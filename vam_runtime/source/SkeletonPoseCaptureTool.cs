// SkeletonPoseCaptureTool.cs
// VaM Timeline AI manual pose ground-truth capture tool.
//
// Read-only by design:
// - does not move controllers
// - does not animate controllers
// - does not save or modify the scene
// - does not create Timeline clips
//
// It draws a lightweight skeleton/debug overlay for two selected Person atoms
// and saves current controller transforms plus partner-relative measurements.

using UnityEngine;
using System;
using System.Collections.Generic;
using SimpleJSON;
using MVR.FileManagementSecure;

public class SkeletonPoseCaptureTool : MVRScript
{
    private JSONStorableBool enableOverlay;
    private JSONStorableBool showLabels;
    private JSONStorableBool showPartnerLinks;
    private JSONStorableBool showAlignmentAxis;
    private JSONStorableBool showContactDistances;
    private JSONStorableBool drawOnlySelectedActors;
    private JSONStorableFloat skeletonLineWidth;
    private JSONStorableFloat jointMarkerSize;

    private JSONStorableStringChooser riderAtomChooser;
    private JSONStorableStringChooser partnerAtomChooser;
    private JSONStorableString poseFamily;
    private JSONStorableString poseSubtype;
    private JSONStorableString motionIntent;
    private JSONStorableString humanNotes;
    private JSONStorableString status;

    private readonly string outputDir = "Saves/PluginData/VAMTimelineAI/pose_captures";
    private readonly Dictionary<string, GameObject> labels = new Dictionary<string, GameObject>();
    private readonly Dictionary<string, GameObject> skeletonLines = new Dictionary<string, GameObject>();
    private readonly Dictionary<string, GameObject> jointMarkers = new Dictionary<string, GameObject>();
    private readonly List<string> knownControllerNames = new List<string>() {
        "pelvisControl", "hipControl", "abdomenControl", "abdomen2Control", "chestControl", "neckControl", "headControl",
        "lShoulderControl", "rShoulderControl", "lElbowControl", "rElbowControl", "lHandControl", "rHandControl",
        "lThighControl", "rThighControl", "lKneeControl", "rKneeControl", "lFootControl", "rFootControl", "lToeControl", "rToeControl"
    };

    private readonly List<string> lastMissing = new List<string>();
    private int lastControllerCount = 0;
    private string lastSavedPath = "";

    public override void Init()
    {
        try
        {
            enableOverlay = new JSONStorableBool("Enable Skeleton Overlay", true);
            CreateToggle(enableOverlay);
            showLabels = new JSONStorableBool("Show Labels", false);
            CreateToggle(showLabels);
            showPartnerLinks = new JSONStorableBool("Show Partner Links", false);
            CreateToggle(showPartnerLinks);
            showAlignmentAxis = new JSONStorableBool("Show Alignment Axis", false);
            CreateToggle(showAlignmentAxis);
            showContactDistances = new JSONStorableBool("Show Contact Distances", false);
            CreateToggle(showContactDistances);
            drawOnlySelectedActors = new JSONStorableBool("Draw Only Selected Actors", true);
            CreateToggle(drawOnlySelectedActors);

            skeletonLineWidth = new JSONStorableFloat("Skeleton Line Width", 0.012f, 0.002f, 0.05f);
            CreateSlider(skeletonLineWidth);
            jointMarkerSize = new JSONStorableFloat("Joint Marker Size", 0.035f, 0.005f, 0.12f);
            CreateSlider(jointMarkerSize);

            riderAtomChooser = new JSONStorableStringChooser("Rider / Actor Atom", AtomChoices(), "", "Rider / Actor Atom");
            CreatePopup(riderAtomChooser);
            partnerAtomChooser = new JSONStorableStringChooser("Partner / Receiver Atom", AtomChoices(), "", "Partner / Receiver Atom");
            CreatePopup(partnerAtomChooser);

            poseFamily = new JSONStorableString("Pose Family", "");
            CreateTextField(poseFamily);
            poseSubtype = new JSONStorableString("Pose Subtype", "");
            CreateTextField(poseSubtype);
            motionIntent = new JSONStorableString("Motion Intent / Notes", "");
            CreateTextField(motionIntent);
            humanNotes = new JSONStorableString("Human Notes", "");
            CreateTextField(humanNotes);

            UIDynamicButton refreshButton = CreateButton("Refresh Atoms");
            refreshButton.button.onClick.AddListener(RefreshAtoms);

            UIDynamicButton captureButton = CreateButton("Capture Pose Snapshot");
            captureButton.button.onClick.AddListener(CapturePoseSnapshot);

            UIDynamicButton captureMetaButton = CreateButton("Capture Pose Snapshot + Screenshot Metadata");
            captureMetaButton.button.onClick.AddListener(CapturePoseSnapshotWithScreenshotMetadata);

            UIDynamicButton clearButton = CreateButton("Clear Overlay");
            clearButton.button.onClick.AddListener(ClearOverlay);

            UIDynamicButton openButton = CreateButton("Open Output Folder");
            openButton.button.onClick.AddListener(OpenOutputFolder);

            status = new JSONStorableString("Status", "SkeletonPoseCaptureTool ready. Select rider and partner atoms.");
            CreateTextField(status);
            RefreshAtoms();
        }
        catch (Exception e)
        {
            SuperController.LogError("SkeletonPoseCaptureTool Init error: " + e);
        }
    }

    private void Update()
    {
        if (enableOverlay == null || !enableOverlay.val)
        {
            SetOverlayActive(false);
            return;
        }
        SetOverlayActive(false);
        Atom rider = FindAtom(riderAtomChooser != null ? riderAtomChooser.val : "");
        Atom partner = FindAtom(partnerAtomChooser != null ? partnerAtomChooser.val : "");
        if (rider != null) DrawActorSkeleton(rider, Color.cyan, "rider");
        if (partner != null) DrawActorSkeleton(partner, Color.yellow, "partner");
        if (rider != null && partner != null) DrawInteractionHelpers(rider, partner);
    }

    private void RefreshAtoms()
    {
        List<string> choices = AtomChoices();
        if (riderAtomChooser != null) riderAtomChooser.choices = choices;
        if (partnerAtomChooser != null) partnerAtomChooser.choices = choices;
        SetStatus("Atoms refreshed. Available atoms: " + choices.Count);
    }

    private List<string> AtomChoices()
    {
        List<string> ids = new List<string>();
        try
        {
            List<Atom> atoms = SuperController.singleton.GetAtoms();
            foreach (Atom atom in atoms)
            {
                if (atom == null || string.IsNullOrEmpty(atom.uid)) continue;
                ids.Add(atom.uid);
            }
        }
        catch
        {
            if (containingAtom != null && !string.IsNullOrEmpty(containingAtom.uid)) ids.Add(containingAtom.uid);
        }
        ids.Sort();
        return ids;
    }

    private Atom FindAtom(string uid)
    {
        if (string.IsNullOrEmpty(uid)) return null;
        try { return SuperController.singleton.GetAtomByUid(uid); }
        catch { return null; }
    }

    private void DrawActorSkeleton(Atom atom, Color color, string prefix)
    {
        ControllerMap c = DiscoverControllers(atom);
        DrawJoint(c, "headControl", prefix + "_joint_head", color);
        DrawJoint(c, "chestControl", prefix + "_joint_chest", color);
        DrawJoint(c, "abdomenControl", prefix + "_joint_abdomen", color);
        DrawJoint(c, "abdomen2Control", prefix + "_joint_abdomen2", color);
        DrawJoint(c, "pelvisControl", prefix + "_joint_pelvis", color);
        DrawJoint(c, "lElbowControl", prefix + "_joint_lElbow", color);
        DrawJoint(c, "rElbowControl", prefix + "_joint_rElbow", color);
        DrawJoint(c, "lHandControl", prefix + "_joint_lHand", color);
        DrawJoint(c, "rHandControl", prefix + "_joint_rHand", color);
        DrawJoint(c, "lKneeControl", prefix + "_joint_lKnee", color);
        DrawJoint(c, "rKneeControl", prefix + "_joint_rKnee", color);
        DrawJoint(c, "lFootControl", prefix + "_joint_lFoot", color);
        DrawJoint(c, "rFootControl", prefix + "_joint_rFoot", color);

        if (showLabels != null && showLabels.val)
        {
            Label(c, "headControl", prefix + "_head", "head", color);
            Label(c, "chestControl", prefix + "_chest", prefix == "partner" ? "partner_chest" : "chest", color);
            Label(c, "abdomenControl", prefix + "_abdomen", "abdomen", color);
            Label(c, "pelvisControl", prefix + "_pelvis", prefix == "partner" ? "partner_pelvis" : "pelvis", color);
            Label(c, "lHandControl", prefix + "_lHand", "lHand", color);
            Label(c, "rHandControl", prefix + "_rHand", "rHand", color);
            Label(c, "lKneeControl", prefix + "_lKnee", "lKnee", color);
            Label(c, "rKneeControl", prefix + "_rKnee", "rKnee", color);
            Label(c, "lFootControl", prefix + "_lFoot", "lFoot", color);
            Label(c, "rFootControl", prefix + "_rFoot", "rFoot", color);
        }
    }

    private void DrawInteractionHelpers(Atom rider, Atom partner)
    {
        ControllerMap r = DiscoverControllers(rider);
        ControllerMap p = DiscoverControllers(partner);
        FreeControllerV3 rp = r.Get("pelvisControl");
        FreeControllerV3 pp = p.Get("pelvisControl");
        FreeControllerV3 rh = r.Get("headControl");
        FreeControllerV3 pc = p.Get("chestControl");

        if (showPartnerLinks != null && showPartnerLinks.val)
        {
            DrawLine(rp, pp, Color.magenta, "helper_rider_pelvis_partner_pelvis");
            DrawLine(rh, pp, Color.green, "helper_rider_head_partner_pelvis");
        }
        if (showContactDistances != null && showContactDistances.val)
        {
            DrawLine(r.Get("lHandControl"), pc, Color.white, "helper_lHand_partner_chest");
            DrawLine(r.Get("rHandControl"), pc, Color.white, "helper_rHand_partner_chest");
            DrawLine(r.Get("lHandControl"), pp, Color.gray, "helper_lHand_partner_pelvis");
            DrawLine(r.Get("rHandControl"), pp, Color.gray, "helper_rHand_partner_pelvis");
            DrawLine(r.Get("lHandControl"), p.Get("lThighControl"), Color.red, "helper_lHand_partner_lThigh");
            DrawLine(r.Get("rHandControl"), p.Get("rThighControl"), Color.red, "helper_rHand_partner_rThigh");
        }
        if (showAlignmentAxis != null && showAlignmentAxis.val && pp != null)
        {
            Transform t = pp.transform;
            SetLine("axis_partner_forward", t.position, t.position + t.forward * 0.45f, Color.blue);
            SetLine("axis_partner_up", t.position, t.position + t.up * 0.45f, Color.green);
            SetLine("axis_partner_right", t.position, t.position + t.right * 0.45f, Color.red);
        }
    }

    private void DrawBone(ControllerMap c, string a, string b, Color color, string key)
    {
        FreeControllerV3 ca = c.Get(a);
        FreeControllerV3 cb = c.Get(b);
        DrawLine(ca, cb, color, "bone_" + key);
    }

    private void DrawLine(FreeControllerV3 a, FreeControllerV3 b, Color color, string key)
    {
        if (a == null || b == null) return;
        SetLine(key, a.transform.position, b.transform.position, color);
    }

    private void SetLine(string key, Vector3 a, Vector3 b, Color color)
    {
        GameObject go = GetLineObject(key, color);
        go.SetActive(true);
        LineRenderer lr = go.GetComponent<LineRenderer>();
        float w = skeletonLineWidth != null ? skeletonLineWidth.val : 0.012f;
        lr.startWidth = w;
        lr.endWidth = w;
        lr.SetPosition(0, a);
        lr.SetPosition(1, b);
    }

    private GameObject GetLineObject(string key, Color color)
    {
        GameObject go;
        if (skeletonLines.TryGetValue(key, out go) && go != null) return go;
        go = new GameObject("VAMTimelineAI_ESPLine_" + key);
        LineRenderer lr = go.AddComponent<LineRenderer>();
        lr.useWorldSpace = true;
        lr.positionCount = 2;
        lr.startWidth = 0.012f;
        lr.endWidth = 0.012f;
        lr.numCapVertices = 4;
        lr.material = MakeMaterial(color);
        lr.startColor = color;
        lr.endColor = color;
        skeletonLines[key] = go;
        return go;
    }

    private void DrawJoint(ControllerMap c, string controllerName, string key, Color color)
    {
        FreeControllerV3 fc = c.Get(controllerName);
        if (fc == null) return;
        GameObject marker = GetJointMarker(key, color);
        marker.SetActive(true);
        marker.transform.position = fc.transform.position;
        float s = jointMarkerSize != null ? jointMarkerSize.val : 0.035f;
        marker.transform.localScale = new Vector3(s, s, s);
    }

    private GameObject GetJointMarker(string key, Color color)
    {
        GameObject go;
        if (jointMarkers.TryGetValue(key, out go) && go != null) return go;
        go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        go.name = "VAMTimelineAI_ESPJoint_" + key;
        Collider col = go.GetComponent<Collider>();
        if (col != null) Destroy(col);
        Renderer r = go.GetComponent<Renderer>();
        if (r != null) r.material = MakeMaterial(color);
        jointMarkers[key] = go;
        return go;
    }

    private Material MakeMaterial(Color color)
    {
        Shader shader = Shader.Find("Unlit/Color");
        if (shader == null) shader = Shader.Find("Sprites/Default");
        Material mat = new Material(shader);
        mat.color = color;
        return mat;
    }

    private void Label(ControllerMap c, string controllerName, string key, string text, Color color)
    {
        FreeControllerV3 fc = c.Get(controllerName);
        if (fc == null) return;
        GameObject label = GetLabel(key);
        label.SetActive(true);
        label.transform.position = fc.transform.position + new Vector3(0f, 0.035f, 0f);
        TextMesh tm = label.GetComponent<TextMesh>();
        tm.text = text;
        tm.color = color;
    }

    private GameObject GetLabel(string key)
    {
        GameObject go;
        if (labels.TryGetValue(key, out go) && go != null) return go;
        go = new GameObject("VAMTimelineAI_Label_" + key);
        TextMesh tm = go.AddComponent<TextMesh>();
        tm.fontSize = 32;
        tm.characterSize = 0.018f;
        tm.anchor = TextAnchor.MiddleCenter;
        labels[key] = go;
        return go;
    }

    private void ClearOverlay()
    {
        List<string> keys = new List<string>(labels.Keys);
        for (int i = 0; i < keys.Count; i++)
        {
            GameObject go = null;
            if (labels.TryGetValue(keys[i], out go) && go != null) Destroy(go);
        }
        labels.Clear();

        keys = new List<string>(skeletonLines.Keys);
        for (int i = 0; i < keys.Count; i++)
        {
            GameObject go = null;
            if (skeletonLines.TryGetValue(keys[i], out go) && go != null) Destroy(go);
        }
        skeletonLines.Clear();

        keys = new List<string>(jointMarkers.Keys);
        for (int i = 0; i < keys.Count; i++)
        {
            GameObject go = null;
            if (jointMarkers.TryGetValue(keys[i], out go) && go != null) Destroy(go);
        }
        jointMarkers.Clear();
        SetStatus("Overlay cleared.");
    }

    private void SetOverlayActive(bool active)
    {
        List<string> keys = new List<string>(labels.Keys);
        for (int i = 0; i < keys.Count; i++)
        {
            GameObject go = null;
            if (labels.TryGetValue(keys[i], out go) && go != null) go.SetActive(active);
        }
        keys = new List<string>(skeletonLines.Keys);
        for (int i = 0; i < keys.Count; i++)
        {
            GameObject go = null;
            if (skeletonLines.TryGetValue(keys[i], out go) && go != null) go.SetActive(active);
        }
        keys = new List<string>(jointMarkers.Keys);
        for (int i = 0; i < keys.Count; i++)
        {
            GameObject go = null;
            if (jointMarkers.TryGetValue(keys[i], out go) && go != null) go.SetActive(active);
        }
    }

    private void CapturePoseSnapshot()
    {
        CapturePoseSnapshotInternal(false);
    }

    private void CapturePoseSnapshotWithScreenshotMetadata()
    {
        CapturePoseSnapshotInternal(true);
    }

    private void CapturePoseSnapshotInternal(bool includeScreenshotMetadata)
    {
        Atom rider = FindAtom(riderAtomChooser != null ? riderAtomChooser.val : "");
        Atom partner = FindAtom(partnerAtomChooser != null ? partnerAtomChooser.val : "");
        JSONClass root = new JSONClass();
        root["schema_version"] = "pose_capture_v1";
        root["created_at"] = DateTime.UtcNow.ToString("o");
        root["source"] = "VaM SkeletonPoseCaptureTool";
        root["vam_version"] = Application.version;
        root["scene_name"] = SuperController.singleton != null ? SuperController.singleton.currentLoadDir : "";

        JSONClass human = new JSONClass();
        human["pose_family"] = poseFamily != null ? poseFamily.val : "unknown";
        human["pose_subtype"] = poseSubtype != null ? poseSubtype.val : "unknown";
        human["motion_intent"] = motionIntent != null ? motionIntent.val : "";
        human["human_notes"] = humanNotes != null ? humanNotes.val : "";
        root["human_labels"] = human;

        JSONClass atoms = new JSONClass();
        atoms["rider"] = CaptureAtom(rider, "rider");
        atoms["partner"] = CaptureAtom(partner, "partner");
        root["atoms"] = atoms;
        root["derived"] = BuildDerived(rider, partner);
        root["pose_quality_flags"] = BuildQualityFlags(rider, partner);
        if (includeScreenshotMetadata) root["screenshot_metadata"] = BuildScreenshotMetadata();

        string fileName = "pose_capture_" + DateTime.Now.ToString("yyyyMMdd_HHmmss") + ".json";
        string vamPath = outputDir + "/" + fileName;
        try
        {
            string json = root.ToString();
            FileManagerSecure.CreateDirectory(outputDir);
            FileManagerSecure.WriteAllText(vamPath, json);
            lastSavedPath = vamPath;
            SetStatus("Saved pose snapshot: " + vamPath);
        }
        catch (Exception e)
        {
            SetStatus("Save failed: " + e.Message);
            SuperController.LogError("SkeletonPoseCaptureTool save error: " + e);
        }
    }

    private JSONClass CaptureAtom(Atom atom, string role)
    {
        JSONClass data = new JSONClass();
        data["atom_uid"] = atom != null ? atom.uid : "";
        data["atom_name"] = atom != null ? atom.name : "";
        JSONClass controllers = new JSONClass();
        JSONArray missing = new JSONArray();
        if (atom != null)
        {
            ControllerMap map = DiscoverControllers(atom);
            foreach (string name in knownControllerNames)
            {
                FreeControllerV3 fc = map.Get(name);
                if (fc == null)
                {
                    missing.Add(name);
                    continue;
                }
                controllers[name] = CaptureController(atom, fc);
            }
            foreach (string name in map.extraControllerNames)
            {
                if (controllers[name] == null) controllers[name] = CaptureController(atom, map.Get(name));
            }
        }
        data["controllers"] = controllers;
        data["missing_controllers"] = missing;
        return data;
    }

    private JSONClass CaptureController(Atom atom, FreeControllerV3 fc)
    {
        JSONClass data = new JSONClass();
        data["exists"].AsBool = fc != null;
        if (fc == null) return data;
        data["world_position"] = Vec(fc.transform.position);
        data["world_rotation_quat"] = Quat(fc.transform.rotation);
        data["local_position_to_atom"] = Vec(atom != null ? atom.transform.InverseTransformPoint(fc.transform.position) : fc.transform.localPosition);
        data["local_rotation_to_atom_quat"] = Quat(atom != null ? Quaternion.Inverse(atom.transform.rotation) * fc.transform.rotation : fc.transform.localRotation);
        data["active"].AsBool = fc.gameObject.activeInHierarchy;
        return data;
    }

    private JSONClass BuildDerived(Atom rider, Atom partner)
    {
        JSONClass derived = new JSONClass();
        ControllerMap r = DiscoverControllers(rider);
        ControllerMap p = DiscoverControllers(partner);
        derived["rider_pelvis_to_partner_pelvis"] = Relation(r.Get("pelvisControl"), p.Get("pelvisControl"), p.Get("pelvisControl"));
        derived["rider_head_to_partner_pelvis"] = Relation(r.Get("headControl"), p.Get("pelvisControl"), p.Get("pelvisControl"));
        derived["rider_lhand_to_partner_chest"] = Relation(r.Get("lHandControl"), p.Get("chestControl"), p.Get("pelvisControl"));
        derived["rider_rhand_to_partner_chest"] = Relation(r.Get("rHandControl"), p.Get("chestControl"), p.Get("pelvisControl"));
        derived["rider_lhand_to_partner_pelvis"] = Relation(r.Get("lHandControl"), p.Get("pelvisControl"), p.Get("pelvisControl"));
        derived["rider_rhand_to_partner_pelvis"] = Relation(r.Get("rHandControl"), p.Get("pelvisControl"), p.Get("pelvisControl"));
        derived["rider_lhand_to_partner_lthigh_or_hip"] = Relation(r.Get("lHandControl"), First(p.Get("lThighControl"), p.Get("hipControl"), p.Get("pelvisControl")), p.Get("pelvisControl"));
        derived["rider_rhand_to_partner_rthigh_or_hip"] = Relation(r.Get("rHandControl"), First(p.Get("rThighControl"), p.Get("hipControl"), p.Get("pelvisControl")), p.Get("pelvisControl"));
        derived["rider_feet_relative_to_pelvis"] = PairRelative(r.Get("lFootControl"), r.Get("rFootControl"), r.Get("pelvisControl"));
        derived["rider_knees_relative_to_pelvis"] = PairRelative(r.Get("lKneeControl"), r.Get("rKneeControl"), r.Get("pelvisControl"));
        derived["partner_local_frame"] = LocalFrame(p.Get("pelvisControl"));
        derived["orientation_hints"] = OrientationHints(r, p);
        return derived;
    }

    private JSONClass Relation(FreeControllerV3 a, FreeControllerV3 b, FreeControllerV3 partnerPelvis)
    {
        JSONClass rel = new JSONClass();
        if (a == null || b == null)
        {
            rel["exists"].AsBool = false;
            return rel;
        }
        Vector3 delta = a.transform.position - b.transform.position;
        rel["exists"].AsBool = true;
        rel["world_delta"] = Vec(delta);
        rel["distance"].AsFloat = delta.magnitude;
        if (partnerPelvis != null) rel["partner_local_delta"] = Vec(partnerPelvis.transform.InverseTransformDirection(delta));
        return rel;
    }

    private JSONClass PairRelative(FreeControllerV3 left, FreeControllerV3 right, FreeControllerV3 origin)
    {
        JSONClass data = new JSONClass();
        data["left"] = Relation(left, origin, origin);
        data["right"] = Relation(right, origin, origin);
        return data;
    }

    private JSONClass LocalFrame(FreeControllerV3 origin)
    {
        JSONClass frame = new JSONClass();
        if (origin == null)
        {
            frame["exists"].AsBool = false;
            return frame;
        }
        frame["exists"].AsBool = true;
        frame["origin"] = "partner pelvis";
        frame["world_position"] = Vec(origin.transform.position);
        frame["forward"] = Vec(origin.transform.forward);
        frame["up"] = Vec(origin.transform.up);
        frame["right"] = Vec(origin.transform.right);
        return frame;
    }

    private JSONClass OrientationHints(ControllerMap r, ControllerMap p)
    {
        JSONClass hints = new JSONClass();
        FreeControllerV3 rp = r.Get("pelvisControl");
        FreeControllerV3 rc = r.Get("chestControl");
        FreeControllerV3 rh = r.Get("headControl");
        FreeControllerV3 pp = p.Get("pelvisControl");
        if (rp != null && rc != null) hints["rider_chest_pelvis_vector"] = Vec(rc.transform.position - rp.transform.position);
        if (rc != null && rh != null) hints["rider_head_chest_vector"] = Vec(rh.transform.position - rc.transform.position);
        hints["rider_facing_relative_to_partner"] = "unknown";
        if (rp != null && rc != null && pp != null)
        {
            Vector3 riderForward = rc.transform.forward;
            Vector3 toPartner = (pp.transform.position - rp.transform.position).normalized;
            float dot = Vector3.Dot(riderForward, toPartner);
            if (dot > 0.45f) hints["rider_facing_relative_to_partner"] = "front_to_partner";
            else if (dot < -0.45f) hints["rider_facing_relative_to_partner"] = "back_to_partner";
            else hints["rider_facing_relative_to_partner"] = "side_or_unknown";
            hints["rider_facing_dot_to_partner"].AsFloat = dot;
        }
        hints["pose_hint"] = PoseHint(r);
        return hints;
    }

    private string PoseHint(ControllerMap r)
    {
        FreeControllerV3 pelvis = r.Get("pelvisControl");
        FreeControllerV3 lk = r.Get("lKneeControl");
        FreeControllerV3 rk = r.Get("rKneeControl");
        FreeControllerV3 lf = r.Get("lFootControl");
        FreeControllerV3 rf = r.Get("rFootControl");
        FreeControllerV3 lh = r.Get("lHandControl");
        FreeControllerV3 rh = r.Get("rHandControl");
        FreeControllerV3 chest = r.Get("chestControl");
        if (pelvis == null || chest == null) return "unknown";
        float kneeY = AvgY(lk, rk);
        float footY = AvgY(lf, rf);
        float handY = AvgY(lh, rh);
        if (kneeY < pelvis.transform.position.y - 0.15f && footY < pelvis.transform.position.y - 0.10f) return "kneeling_or_squat";
        if (handY < chest.transform.position.y - 0.25f && chest.transform.position.y < pelvis.transform.position.y + 0.45f) return "all_fours_or_forward_support";
        if (chest.transform.position.y < pelvis.transform.position.y + 0.20f) return "supine_prone_or_low";
        return "standing_or_upright";
    }

    private float AvgY(FreeControllerV3 a, FreeControllerV3 b)
    {
        int n = 0;
        float total = 0f;
        if (a != null) { total += a.transform.position.y; n++; }
        if (b != null) { total += b.transform.position.y; n++; }
        return n > 0 ? total / n : 999f;
    }

    private JSONClass BuildQualityFlags(Atom rider, Atom partner)
    {
        ControllerMap r = DiscoverControllers(rider);
        ControllerMap p = DiscoverControllers(partner);
        JSONClass flags = new JSONClass();
        flags["has_rider"].AsBool = rider != null;
        flags["has_partner"].AsBool = partner != null;
        flags["has_rider_pelvis"].AsBool = r.Get("pelvisControl") != null;
        flags["has_partner_pelvis"].AsBool = p.Get("pelvisControl") != null;
        flags["has_head_chest_pelvis_chain"].AsBool = r.Get("headControl") != null && r.Get("chestControl") != null && r.Get("pelvisControl") != null;
        flags["has_leg_anchors"].AsBool = r.Get("lFootControl") != null && r.Get("rFootControl") != null && r.Get("lKneeControl") != null && r.Get("rKneeControl") != null;
        flags["has_hand_targets"].AsBool = r.Get("lHandControl") != null && r.Get("rHandControl") != null;
        JSONArray warnings = new JSONArray();
        if (rider == null) warnings.Add("missing_rider_atom");
        if (partner == null) warnings.Add("missing_partner_atom");
        if (r.Get("pelvisControl") == null) warnings.Add("missing_rider_pelvisControl");
        if (p.Get("pelvisControl") == null) warnings.Add("missing_partner_pelvisControl");
        flags["warnings"] = warnings;
        return flags;
    }

    private JSONClass BuildScreenshotMetadata()
    {
        JSONClass meta = new JSONClass();
        meta["requested"].AsBool = true;
        meta["screenshot_captured"].AsBool = false;
        meta["note"] = "V1 records metadata only. Use manual screenshot capture if needed.";
        if (Camera.main != null)
        {
            meta["camera_world_position"] = Vec(Camera.main.transform.position);
            meta["camera_world_rotation_quat"] = Quat(Camera.main.transform.rotation);
        }
        return meta;
    }

    private ControllerMap DiscoverControllers(Atom atom)
    {
        ControllerMap map = new ControllerMap();
        if (atom == null) return map;
        foreach (string name in knownControllerNames)
        {
            FreeControllerV3 fc = atom.GetStorableByID(name) as FreeControllerV3;
            if (fc != null) map.controllers[name] = fc;
        }
        try
        {
            List<string> ids = atom.GetStorableIDs();
            foreach (string id in ids)
            {
                if (string.IsNullOrEmpty(id)) continue;
                FreeControllerV3 fc = atom.GetStorableByID(id) as FreeControllerV3;
                if (fc == null) continue;
                if (!map.controllers.ContainsKey(id))
                {
                    map.controllers[id] = fc;
                    map.extraControllerNames.Add(id);
                }
            }
        }
        catch { }
        lastControllerCount = map.controllers.Count;
        lastMissing.Clear();
        foreach (string name in knownControllerNames) if (!map.controllers.ContainsKey(name)) lastMissing.Add(name);
        return map;
    }

    private FreeControllerV3 First(FreeControllerV3 a, FreeControllerV3 b, FreeControllerV3 c)
    {
        if (a != null) return a;
        if (b != null) return b;
        return c;
    }

    private JSONArray Vec(Vector3 v)
    {
        JSONArray arr = new JSONArray();
        arr.Add(Math.Round(v.x, 6).ToString());
        arr.Add(Math.Round(v.y, 6).ToString());
        arr.Add(Math.Round(v.z, 6).ToString());
        return arr;
    }

    private JSONArray Quat(Quaternion q)
    {
        JSONArray arr = new JSONArray();
        arr.Add(Math.Round(q.x, 6).ToString());
        arr.Add(Math.Round(q.y, 6).ToString());
        arr.Add(Math.Round(q.z, 6).ToString());
        arr.Add(Math.Round(q.w, 6).ToString());
        return arr;
    }

    private string SafeName(string text)
    {
        string value = string.IsNullOrEmpty(text) ? "unknown" : text.Trim().ToLowerInvariant();
        string bad = "<>:\"/\\|?*";
        for (int i = 0; i < bad.Length; i++) value = value.Replace(bad[i], '_');
        value = value.Replace(" ", "_");
        return value;
    }

    private void OpenOutputFolder()
    {
        SetStatus("Output folder: " + outputDir + " | Last saved: " + lastSavedPath);
    }

    private void SetStatus(string message)
    {
        string selected = "rider=" + (riderAtomChooser != null ? riderAtomChooser.val : "") + ", partner=" + (partnerAtomChooser != null ? partnerAtomChooser.val : "");
        string controllerText = " controllers=" + lastControllerCount + ", missing=" + lastMissing.Count;
        if (status != null) status.val = message + "\n" + selected + controllerText + "\nlast_saved=" + lastSavedPath;
        SuperController.LogMessage("SkeletonPoseCaptureTool: " + message);
    }

    private void OnDestroy()
    {
        ClearOverlay();
    }

    private class ControllerMap
    {
        public readonly Dictionary<string, FreeControllerV3> controllers = new Dictionary<string, FreeControllerV3>();
        public readonly List<string> extraControllerNames = new List<string>();

        public FreeControllerV3 Get(string name)
        {
            FreeControllerV3 fc;
            return controllers.TryGetValue(name, out fc) ? fc : null;
        }
    }
}
