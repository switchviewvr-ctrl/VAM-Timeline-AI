// GeneratedMotionReviewPlayer.cs
// Review-only VaM plugin for VAM Timeline AI generated relative motion flows.
// It applies controller deltas relative to the current controller positions captured as playback baseline.

using UnityEngine;
using System;
using System.Collections.Generic;
using SimpleJSON;
using MVR.FileManagementSecure;

public class GeneratedMotionReviewPlayer : MVRScript
{
    private JSONStorableString jsonPath;
    private JSONStorableString status;
    private JSONStorableBool loop;
    private JSONStorableBool applyPelvis;
    private JSONStorableBool applyFollowers;
    private JSONStorableBool applyAnchors;
    private JSONStorableFloat playbackSpeed;
    private JSONStorableFloat motionScale;
    private JSONStorableFloat pelvisScale;
    private JSONStorableFloat verticalScale;
    private JSONStorableFloat lateralScale;
    private JSONStorableFloat forwardBackScale;
    private JSONStorableFloat chestFollowerScale;

    private readonly HashSet<string> allowedControllers = new HashSet<string>() {
        "hipControl", "pelvisControl", "abdomenControl", "abdomen2Control", "chestControl", "headControl",
        "lHandControl", "rHandControl", "lElbowControl", "rElbowControl", "lKneeControl", "rKneeControl",
        "lFootControl", "rFootControl", "lThighControl", "rThighControl"
    };

    private class Track
    {
        public string name;
        public string role;
        public FreeControllerV3 controller;
        public List<float> times = new List<float>();
        public List<Vector3> deltas = new List<Vector3>();
        public Vector3 baselinePosition;
        public Quaternion baselineRotation;
    }

    private readonly List<Track> tracks = new List<Track>();
    private bool baselineCaptured = false;
    private bool playing = false;
    private float playTime = 0f;
    private float duration = 0f;
    private string loadedSchema = "";
    private int skippedControllers = 0;
    private int missingControllers = 0;

    public override void Init()
    {
        try
        {
            jsonPath = new JSONStorableString("JSON File Path", "Saves/PluginData/VAMTimelineAI/generated_motion_review_player_v1.json");
            CreateTextField(jsonPath);

            UIDynamicButton loadButton = CreateButton("Load JSON");
            loadButton.button.onClick.AddListener(LoadJson);

            UIDynamicButton baselineButton = CreateButton("Capture Baseline");
            baselineButton.button.onClick.AddListener(CaptureBaseline);

            UIDynamicButton playButton = CreateButton("Play");
            playButton.button.onClick.AddListener(Play);

            UIDynamicButton stopButton = CreateButton("Stop");
            stopButton.button.onClick.AddListener(Stop);

            UIDynamicButton resetButton = CreateButton("Reset To Baseline");
            resetButton.button.onClick.AddListener(ResetToBaseline);

            loop = new JSONStorableBool("Loop", true);
            CreateToggle(loop);

            playbackSpeed = new JSONStorableFloat("Playback Speed", 1.0f, 0.1f, 3.0f);
            CreateSlider(playbackSpeed);

            motionScale = new JSONStorableFloat("Motion Scale", 1.0f, 0.0f, 2.0f);
            CreateSlider(motionScale);

            pelvisScale = new JSONStorableFloat("Pelvis Scale", 1.0f, 0.0f, 2.0f);
            CreateSlider(pelvisScale);
            verticalScale = new JSONStorableFloat("Vertical Scale", 1.0f, 0.0f, 2.0f);
            CreateSlider(verticalScale);
            lateralScale = new JSONStorableFloat("Lateral Scale", 1.0f, 0.0f, 2.0f);
            CreateSlider(lateralScale);
            forwardBackScale = new JSONStorableFloat("Forward/Back Scale", 1.0f, 0.0f, 2.0f);
            CreateSlider(forwardBackScale);
            chestFollowerScale = new JSONStorableFloat("Chest Follower Scale", 1.0f, 0.0f, 2.0f);
            CreateSlider(chestFollowerScale);

            applyPelvis = new JSONStorableBool("Apply Pelvis", true);
            CreateToggle(applyPelvis);
            applyFollowers = new JSONStorableBool("Apply Chest/Abdomen", true);
            CreateToggle(applyFollowers);
            applyAnchors = new JSONStorableBool("Apply Anchors", true);
            CreateToggle(applyAnchors);

            status = new JSONStorableString("Status", "Review-only player ready. Load JSON.");
            CreateTextField(status);
        }
        catch (Exception e)
        {
            SuperController.LogError("GeneratedMotionReviewPlayer Init error: " + e);
        }
    }

    private void LoadJson()
    {
        Stop();
        tracks.Clear();
        baselineCaptured = false;
        duration = 0f;
        skippedControllers = 0;
        missingControllers = 0;
        loadedSchema = "";

        string path = ResolveSecurePath(jsonPath.val);
        if (string.IsNullOrEmpty(path) || !FileManagerSecure.FileExists(path))
        {
            SetStatus("JSON file not found. Use a VaM-relative path like Saves/PluginData/VAMTimelineAI/generated_motion_review_player_v1.json");
            return;
        }

        try
        {
            string text = FileManagerSecure.ReadAllText(path);
            JSONNode root = JSON.Parse(text);
            loadedSchema = root["schema"] != null ? root["schema"].Value : "unknown";
            JSONArray controllers = FindControllerArray(root);
            if (controllers == null)
            {
                SetStatus("No controller track array found in JSON.");
                return;
            }

            for (int i = 0; i < controllers.Count; i++)
            {
                JSONNode node = controllers[i];
                string name = ReadName(node);
                if (!IsAllowedController(name))
                {
                    skippedControllers++;
                    continue;
                }

                FreeControllerV3 fc = containingAtom.GetStorableByID(name) as FreeControllerV3;
                if (fc == null)
                {
                    missingControllers++;
                    SuperController.LogMessage("GeneratedMotionReviewPlayer: missing controller skipped: " + name);
                    continue;
                }

                Track t = new Track();
                t.name = name;
                t.role = node["role"] != null ? node["role"].Value : "unknown";
                t.controller = fc;
                ReadTimes(node, t.times);
                ReadDeltas(node, t.deltas);
                if (t.times.Count == 0 || t.deltas.Count == 0)
                {
                    skippedControllers++;
                    continue;
                }
                duration = Mathf.Max(duration, t.times[t.times.Count - 1]);
                tracks.Add(t);
            }

            SetStatus("Loaded schema " + loadedSchema + ". Tracks " + tracks.Count + ", skipped " + skippedControllers + ", missing " + missingControllers + ". Capture Baseline next.");
        }
        catch (Exception e)
        {
            SetStatus("Load error: " + e.Message);
            SuperController.LogError("GeneratedMotionReviewPlayer LoadJson error: " + e);
        }
    }

    private string ResolveSecurePath(string raw)
    {
        if (string.IsNullOrEmpty(raw)) return raw;
        string p = raw.Replace("\\", "/").Trim();
        if (FileManagerSecure.FileExists(p)) return p;
        int saves = p.IndexOf("Saves/", StringComparison.OrdinalIgnoreCase);
        if (saves >= 0)
        {
            string rel = p.Substring(saves);
            if (FileManagerSecure.FileExists(rel)) return rel;
        }
        if (!p.Contains("/"))
        {
            string pluginData = "Saves/PluginData/VAMTimelineAI/" + p;
            if (FileManagerSecure.FileExists(pluginData)) return pluginData;
        }
        return p;
    }

    private JSONArray FindControllerArray(JSONNode root)
    {
        if (root == null) return null;
        JSONNode controllersNode = root["controllers"];
        if (controllersNode != null)
        {
            JSONArray arr = controllersNode.AsArray;
            if (arr != null) return arr;
        }
        JSONNode tracksNode = root["controller_tracks"];
        if (tracksNode != null)
        {
            JSONArray arr = tracksNode.AsArray;
            if (arr != null) return arr;
        }
        return null;
    }

    private string ReadName(JSONNode node)
    {
        if (node["name"] != null) return node["name"].Value;
        if (node["controller_name"] != null) return node["controller_name"].Value;
        return "";
    }

    private bool IsAllowedController(string name)
    {
        if (string.IsNullOrEmpty(name)) return false;
        string lower = name.ToLowerInvariant();
        if (lower.Contains("person") || lower.Contains("root") || lower.Contains("world") || lower.Contains("atom")) return false;
        return allowedControllers.Contains(name);
    }

    private void ReadTimes(JSONNode node, List<float> times)
    {
        JSONArray arr = node["times"] != null ? node["times"].AsArray : null;
        if (arr == null) return;
        for (int i = 0; i < arr.Count; i++) times.Add(arr[i].AsFloat);
    }

    private void ReadDeltas(JSONNode node, List<Vector3> deltas)
    {
        JSONNode src = node["position_deltas"];
        if (src == null) src = node["position_deltas_applied"];
        if (src == null && node["retargeted_positions"] != null && node["baseline_position"] != null)
        {
            JSONArray baseArr = node["baseline_position"].AsArray;
            Vector3 b = new Vector3(baseArr[0].AsFloat, baseArr[1].AsFloat, baseArr[2].AsFloat);
            JSONArray positions = node["retargeted_positions"].AsArray;
            for (int i = 0; i < positions.Count; i++)
            {
                JSONArray p = positions[i].AsArray;
                deltas.Add(new Vector3(p[0].AsFloat, p[1].AsFloat, p[2].AsFloat) - b);
            }
            return;
        }
        if (src == null) return;
        JSONArray arr = src.AsArray;
        for (int i = 0; i < arr.Count; i++)
        {
            JSONArray p = arr[i].AsArray;
            deltas.Add(new Vector3(p[0].AsFloat, p[1].AsFloat, p[2].AsFloat));
        }
    }

    private void CaptureBaseline()
    {
        foreach (Track t in tracks)
        {
            if (t.controller == null) continue;
            t.baselinePosition = t.controller.transform.position;
            t.baselineRotation = t.controller.transform.rotation;
        }
        baselineCaptured = true;
        SetStatus("Baseline captured for " + tracks.Count + " tracks.");
    }

    private void Play()
    {
        if (tracks.Count == 0)
        {
            SetStatus("Load JSON first.");
            return;
        }
        if (!baselineCaptured) CaptureBaseline();
        playTime = 0f;
        playing = true;
        SetStatus("Playing " + loadedSchema + ". Tracks " + tracks.Count + ". Person/root is never targeted.");
    }

    private void Stop()
    {
        playing = false;
        SetStatus("Stopped.");
    }

    private void ResetToBaseline()
    {
        if (!baselineCaptured)
        {
            SetStatus("No baseline captured.");
            return;
        }
        foreach (Track t in tracks)
        {
            if (t.controller == null) continue;
            t.controller.transform.position = t.baselinePosition;
            t.controller.transform.rotation = t.baselineRotation;
        }
        playing = false;
        playTime = 0f;
        SetStatus("Reset to captured baseline.");
    }

    private void Update()
    {
        if (!playing || tracks.Count == 0 || duration <= 0f) return;
        playTime += Time.deltaTime * playbackSpeed.val;
        if (playTime > duration)
        {
            if (loop.val) playTime = playTime % duration;
            else
            {
                playing = false;
                return;
            }
        }
        ApplyAtTime(playTime);
        if (status != null && Time.frameCount % 30 == 0)
        {
            status.val = "Playing " + loadedSchema + " t=" + playTime.ToString("0.00") + "/" + duration.ToString("0.00") + " tracks=" + tracks.Count + " skipped=" + skippedControllers + " missing=" + missingControllers;
        }
    }

    private void ApplyAtTime(float t)
    {
        foreach (Track track in tracks)
        {
            if (track.controller == null || track.times.Count == 0 || track.deltas.Count == 0) continue;
            if (track.role == "driver" && !applyPelvis.val) continue;
            if (track.role == "follower" && !applyFollowers.val) continue;
            if ((track.role == "anchor" || track.role == "support") && !applyAnchors.val) continue;
            Vector3 delta = SampleDelta(track, t);
            delta = ScaleDelta(track, delta);
            track.controller.transform.position = track.baselinePosition + delta;
        }
    }

    private Vector3 ScaleDelta(Track track, Vector3 delta)
    {
        float roleScale = 1.0f;
        if (track.role == "driver") roleScale = pelvisScale.val;
        else if (track.role == "follower") roleScale = chestFollowerScale.val;
        else if (track.role == "anchor" || track.role == "support") roleScale = applyAnchors.val ? 1.0f : 0.0f;
        return new Vector3(
            delta.x * motionScale.val * roleScale * lateralScale.val,
            delta.y * motionScale.val * roleScale * verticalScale.val,
            delta.z * motionScale.val * roleScale * forwardBackScale.val
        );
    }

    private Vector3 SampleDelta(Track track, float t)
    {
        if (t <= track.times[0]) return track.deltas[0];
        int last = Mathf.Min(track.times.Count, track.deltas.Count) - 1;
        if (t >= track.times[last]) return track.deltas[last];
        for (int i = 0; i < last; i++)
        {
            if (track.times[i + 1] >= t)
            {
                float span = Mathf.Max(0.0001f, track.times[i + 1] - track.times[i]);
                float a = Mathf.Clamp01((t - track.times[i]) / span);
                return Vector3.Lerp(track.deltas[i], track.deltas[i + 1], a);
            }
        }
        return track.deltas[last];
    }

    private void SetStatus(string message)
    {
        if (status != null) status.val = message;
        SuperController.LogMessage("GeneratedMotionReviewPlayer: " + message);
    }

    public void OnDisable()
    {
        Stop();
    }
}
