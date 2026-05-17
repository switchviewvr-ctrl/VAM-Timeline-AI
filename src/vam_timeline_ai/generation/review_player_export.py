"""Export generated/retargeted flows for the VaM review player."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import shutil

from vam_timeline_ai.generation.generated_motion import ALLOWED_GENERATED_CONTROLLERS
from vam_timeline_ai.io.json_utils import dump_json, load_json


SCRIPT_SOURCE = Path("vam_runtime/source/GeneratedMotionReviewPlayer.cs")
DEFAULT_VAM_SCRIPT_PATH = Path("G:/VAM/Custom/Scripts/VAMTimelineAI/GeneratedMotionReviewPlayer.cs")
DEFAULT_VAM_JSON_PATH = Path("G:/VAM/Saves/PluginData/VAMTimelineAI/generated_motion_review_player_v0.json")
DEFAULT_VAM_SECURE_JSON_PATH = "Saves/PluginData/VAMTimelineAI/generated_motion_review_player_v0.json"
DEFAULT_VAM_JSON_PATH_V1 = Path("G:/VAM/Saves/PluginData/VAMTimelineAI/generated_motion_review_player_v1.json")
DEFAULT_VAM_SECURE_JSON_PATH_V1 = "Saves/PluginData/VAMTimelineAI/generated_motion_review_player_v1.json"


def export_generated_flow_for_vam_review(retargeted_flow: str | Path, out: str | Path, report: str | Path) -> dict[str, Any]:
    source = load_json(retargeted_flow)
    controllers = []
    skipped = []
    max_duration = 0.0
    fps = 60.0
    for track in source.get("controller_tracks", []) or []:
        name = str(track.get("controller_name") or track.get("name") or "")
        if name not in ALLOWED_GENERATED_CONTROLLERS:
            skipped.append({"name": name, "reason": "disallowed_controller"})
            continue
        times = [float(v) for v in (track.get("times") or [])]
        if len(times) > 1:
            max_duration = max(max_duration, max(times))
            dt = times[1] - times[0]
            if dt > 0:
                fps = round(1.0 / dt, 6)
        deltas = track.get("position_deltas_applied")
        if deltas is None:
            deltas = track.get("position_deltas")
        if deltas is None and track.get("retargeted_positions") and track.get("baseline_position"):
            base = [float(v) for v in track.get("baseline_position")]
            deltas = [[round(float(p[i]) - base[i], 6) for i in range(3)] for p in track.get("retargeted_positions")]
        if not deltas:
            skipped.append({"name": name, "reason": "missing_position_deltas"})
            continue
        controllers.append({
            "name": name,
            "bodypart": track.get("bodypart"),
            "role": track.get("role") or "unknown",
            "times": times,
            "position_deltas": deltas,
            "rotation_deltas": track.get("rotation_deltas") or None,
        })
    data = {
        "schema": "vam_generated_motion_review_player_v0",
        "review_only": True,
        "coordinate_mode": "relative_to_playback_baseline",
        "source_world_coords_used": False,
        "person_root_tracks_included": False,
        "clip_stitching_used": False,
        "native_timeline_importable": False,
        "fps": fps,
        "duration_seconds": round(max_duration, 6),
        "controllers": controllers,
        "safety": {
            "allowed_controllers": sorted(ALLOWED_GENERATED_CONTROLLERS),
            "apply_as_relative_deltas_from_current_baseline": True,
            "skip_disallowed_controllers": True,
            "never_move_person_root": True,
            "review_only": True,
        },
        "source_flow_id": source.get("flow_id"),
        "source_schema": source.get("schema"),
        "skipped_tracks": skipped,
        "warnings": [
            "Load this file with GeneratedMotionReviewPlayer.cs, not VaM Timeline import.",
            "The player applies deltas from current controller positions captured at playback baseline.",
        ],
    }
    dump_json(out, data)
    _write_export_report(data, report)
    return data


def export_generated_flow_for_vam_review_v1(retargeted_flow: str | Path, out: str | Path, report: str | Path) -> dict[str, Any]:
    data = export_generated_flow_for_vam_review(retargeted_flow, out, report)
    source = load_json(retargeted_flow)
    data.update({
        "schema": "vam_generated_motion_review_player_v1",
        "baseline_style": source.get("baseline_style") or "cowgirl_kneeling_forward",
        "coordination_profile": source.get("coordination_profile") or "cowgirl_oval_grind_v1",
        "axis_scales": source.get("axis_scales", {}),
        "follower_info": {
            "abdomen_chest_head_follow_pelvis": True,
            "followers_are_damped": True,
        },
        "anchors": [c.get("name") for c in data.get("controllers", []) if c.get("role") in {"anchor", "support"}],
    })
    dump_json(out, data)
    _write_export_report(data, report)
    return data


def prepare_vam_review_player_v0(retargeted_flow: str | Path, out_dir: str | Path) -> dict[str, Any]:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_out = target / "generated_motion_review_player_v0.json"
    report = target / "generated_motion_review_player_v0_report.md"
    data = export_generated_flow_for_vam_review(retargeted_flow, json_out, report)
    script_source = write_review_player_script(SCRIPT_SOURCE)
    copied_to = None
    if DEFAULT_VAM_SCRIPT_PATH.parent.exists() or DEFAULT_VAM_SCRIPT_PATH.parent.parent.exists():
        DEFAULT_VAM_SCRIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(script_source, DEFAULT_VAM_SCRIPT_PATH)
        copied_to = str(DEFAULT_VAM_SCRIPT_PATH)
    json_copied_to = None
    if DEFAULT_VAM_JSON_PATH.parent.exists() or DEFAULT_VAM_JSON_PATH.parent.parent.exists():
        DEFAULT_VAM_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(json_out, DEFAULT_VAM_JSON_PATH)
        json_copied_to = str(DEFAULT_VAM_JSON_PATH)
    instructions = target / "VAM_REVIEW_PLAYER_INSTRUCTIONS.md"
    _write_instructions(instructions, json_out, script_source, copied_to, json_copied_to)
    status = Path("data/runs/clean_v2/generation/timeline_export_v0/review_export_status.md")
    if status.parent.exists():
        write_review_export_status(status)
    summary = {
        "schema": "prepare_vam_review_player_v0",
        "review_player_json": str(json_out),
        "script_source": str(script_source),
        "script_copied_to": copied_to,
        "json_copied_to": json_copied_to,
        "vam_secure_json_path": DEFAULT_VAM_SECURE_JSON_PATH,
        "instructions": str(instructions),
        "controller_count": len(data.get("controllers", [])),
        "native_timeline_importable": False,
    }
    dump_json(target / "prepare_vam_review_player_v0_summary.json", summary)
    return summary


def prepare_vam_review_player_v1(retargeted_flow: str | Path, out_dir: str | Path) -> dict[str, Any]:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_out = target / "generated_motion_review_player_v1.json"
    report = target / "generated_motion_review_player_v1_report.md"
    data = export_generated_flow_for_vam_review_v1(retargeted_flow, json_out, report)
    script_source = write_review_player_script(SCRIPT_SOURCE)
    copied_to = None
    if DEFAULT_VAM_SCRIPT_PATH.parent.exists() or DEFAULT_VAM_SCRIPT_PATH.parent.parent.exists():
        DEFAULT_VAM_SCRIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(script_source, DEFAULT_VAM_SCRIPT_PATH)
        copied_to = str(DEFAULT_VAM_SCRIPT_PATH)
    json_copied_to = None
    if DEFAULT_VAM_JSON_PATH_V1.parent.exists() or DEFAULT_VAM_JSON_PATH_V1.parent.parent.exists():
        DEFAULT_VAM_JSON_PATH_V1.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(json_out, DEFAULT_VAM_JSON_PATH_V1)
        json_copied_to = str(DEFAULT_VAM_JSON_PATH_V1)
    instructions = target / "VAM_REVIEW_PLAYER_V1_INSTRUCTIONS.md"
    _write_instructions(instructions, json_out, script_source, copied_to, json_copied_to)
    text = instructions.read_text(encoding="utf-8")
    text = text.replace(DEFAULT_VAM_SECURE_JSON_PATH, DEFAULT_VAM_SECURE_JSON_PATH_V1)
    instructions.write_text(text + "\n## V1 Notes\n\nUse the new axis sliders to reduce lateral hula-hoop motion and increase vertical/forward-back motion during review.\n", encoding="utf-8")
    summary = {
        "schema": "prepare_vam_review_player_v1",
        "review_player_json": str(json_out),
        "script_source": str(script_source),
        "script_copied_to": copied_to,
        "json_copied_to": json_copied_to,
        "vam_secure_json_path": DEFAULT_VAM_SECURE_JSON_PATH_V1,
        "instructions": str(instructions),
        "controller_count": len(data.get("controllers", [])),
        "native_timeline_importable": False,
    }
    dump_json(target / "prepare_vam_review_player_v1_summary.json", summary)
    return summary


def write_review_player_script(path: str | Path = SCRIPT_SOURCE) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(REVIEW_PLAYER_SCRIPT, encoding="utf-8")
    return target


def write_review_export_status(path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "# Review Flow JSON Status\n\n"
        "- native_timeline_importable: false\n"
        "- schema: review_only_retargeted_flow_timeline_v0\n"
        "- current_file: review_only_timeline_v0.json\n"
        "- status: This is not native VaM Timeline plugin JSON and is not importable through Timeline.\n"
        "- recommended_test_method: VaM Generated Motion Review Player\n"
        "- recommended_review_player_json: data/runs/clean_v2/generation/vam_review_player/generated_motion_review_player_v0.json\n",
        encoding="utf-8",
    )


def _write_export_report(data: dict[str, Any], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# VaM Generated Motion Review Player Export V0",
        "",
        "This JSON is for `GeneratedMotionReviewPlayer.cs`. It is not native Timeline JSON.",
        "",
        f"- Schema: `{data.get('schema')}`",
        f"- Native Timeline importable: `{data.get('native_timeline_importable')}`",
        f"- Coordinate mode: `{data.get('coordinate_mode')}`",
        f"- Controllers: `{[c.get('name') for c in data.get('controllers', [])]}`",
        f"- Source world coords used: `{data.get('source_world_coords_used')}`",
        f"- Person/root tracks included: `{data.get('person_root_tracks_included')}`",
        f"- Clip stitching used: `{data.get('clip_stitching_used')}`",
        f"- Skipped tracks: `{data.get('skipped_tracks')}`",
    ]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_instructions(path: Path, json_out: Path, script_source: Path, copied_to: str | None, json_copied_to: str | None) -> None:
    copy_target = copied_to or "<VaM>/Custom/Scripts/VAMTimelineAI/GeneratedMotionReviewPlayer.cs"
    secure_json = DEFAULT_VAM_SECURE_JSON_PATH
    lines = [
        "# VaM Generated Motion Review Player Instructions",
        "",
        "This is a review-only player. It is not Timeline import and not production generation.",
        "",
        "## Install Script",
        "",
        f"1. Script source in this repo: `{script_source}`",
        f"2. Copy to VaM scripts path: `{copy_target}`",
    ]
    if copied_to:
        lines.append("3. The prepare command already copied the script to that VaM path.")
    lines.extend([
        "",
        "## In VaM",
        "",
        "1. Load a simple scene with one Person.",
        "2. Add `GeneratedMotionReviewPlayer.cs` as a plugin on the Person atom.",
        f"3. Set JSON file path to VaM's secure relative path: `{secure_json}`",
        "4. Click `Load JSON`.",
        "5. Click `Capture Baseline` while the Person is in the pose you want to test from.",
        "6. Click `Play`.",
        "",
        "## What To Check",
        "",
        "- Does pelvis move in an oval grind?",
        "- Does the Person atom stay in place?",
        "- Do feet/knees remain stable?",
        "- Is motion too large or too small?",
        "- Does `Reset To Baseline` restore the pose?",
        "",
        "## Warnings",
        "",
        "- Review-only.",
        "- Not Timeline import.",
        "- Not production generator.",
        "- Uses current pose as baseline.",
        "- The script never intentionally targets Person/root/world transforms.",
        "",
        "## File Locations",
        "",
        f"- Project copy: `{json_out.resolve()}`",
        f"- VaM secure copy: `{json_copied_to or 'not copied automatically'}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


REVIEW_PLAYER_SCRIPT = r'''// GeneratedMotionReviewPlayer.cs
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
'''
