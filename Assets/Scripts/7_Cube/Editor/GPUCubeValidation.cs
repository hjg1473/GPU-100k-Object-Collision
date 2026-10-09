using System;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using UnityEditor;
using UnityEngine;
using Object = UnityEngine.Object;

/// <summary>Real GPU regression checks. Run from Tools/GPU Cubes; no scene is saved or replaced.</summary>
[InitializeOnLoad]
public static class GPUCubeValidation
{
    static string gpuError;
    static GPUCubeValidation()
    {
        AssemblyReloadEvents.beforeAssemblyReload += StopSoak;
        EditorApplication.quitting += StopSoak;
        EditorApplication.playModeStateChanged += state => { if (state == PlayModeStateChange.ExitingEditMode) StopSoak(); };
    }
    static void CaptureError(string message, string stack, LogType type)
    {
        if (type == LogType.Error || type == LogType.Exception || type == LogType.Assert) gpuError = message;
    }
    static void BeginErrorCapture() { gpuError = null; Application.logMessageReceived += CaptureError; }
    static void CheckErrors() { Require(gpuError == null, "Unity/GPU error: " + gpuError); }
    const BindingFlags Private = BindingFlags.Instance | BindingFlags.NonPublic;
    static FieldInfo Field(string name) => typeof(GPUCube_gpt).GetField(name, Private);
    static GraphicsBuffer Buffer(GPUCube_gpt sim, string name) => (GraphicsBuffer)Field(name).GetValue(sim);
    static void Set(GPUCube_gpt sim, string name, object value) => Field(name).SetValue(sim, value);
    static void Call(GPUCube_gpt sim, string name) => typeof(GPUCube_gpt).GetMethod(name, Private).Invoke(sim, null);
    static void Require(bool value, string message) { if (!value) throw new Exception(message); }
    static T[] Read<T>(GPUCube_gpt sim, string name, int length) where T : struct
    {
        var result = new T[length]; Buffer(sim, name).GetData(result); return result;
    }
    static GPUCube_gpt Create(int count, int contacts)
    {
        var go = new GameObject("GPU cube validation") { hideFlags = HideFlags.HideAndDontSave };
        go.SetActive(false);
        var sim = go.AddComponent<GPUCube_gpt>();
        sim.computeShader = AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/Scripts/7_Cube/7_Compute.compute");
        sim.mat = new Material(Shader.Find("Custom/7_Cube")) { hideFlags = HideFlags.HideAndDontSave };
        sim.mesh = Resources.GetBuiltinResource<Mesh>("Cube.fbx");
        sim.count = (uint)count;
        Set(sim, "maxContacts", contacts); Set(sim, "diagnostics", false);
        Call(sim, "OnEnable");
        Require(sim.AllocatedBytes > 0, "Initialization failed");
        return sim;
    }
    static void Dispose(GPUCube_gpt sim)
    {
        if (!sim) return;
        Call(sim, "Release");
        Object.DestroyImmediate(sim.mat);
        Object.DestroyImmediate(sim.gameObject);
    }
    static void State(GPUCube_gpt sim, Vector3[] positions, Vector3[] velocities)
    {
        Buffer(sim, "positions").SetData(positions);
        Buffer(sim, "velocities").SetData(velocities);
        Buffer(sim, "angularVelocities").SetData(new Vector3[positions.Length]);
        var rotations = new Vector4[positions.Length];
        for (int i = 0; i < rotations.Length; i++) rotations[i] = new Vector4(0, 0, 0, 1);
        Buffer(sim, "rotations").SetData(rotations);
    }

    [MenuItem("Tools/GPU Cubes/Run GPU Regression Checks")]
    public static void RunRegression()
    {
        Require(!EditorApplication.isPlaying && !soak, "Stop Play mode and any active soak before regression checks.");
        BeginErrorCapture();
        GPUCube_gpt sim = null;
        try
        {
            sim = Create(2, 128);
            // Full grid, padded final block and all three hierarchy levels.
            var counts = new uint[1000000];
            for (int i = 0; i < counts.Length; i++) counts[i] = i % 7 == 0 ? (uint)(i % 5 + 1) : 0;
            Buffer(sim, "cellCounts").SetData(counts);
            Call(sim, "BuildCellStarts");
            var starts = Read<uint>(sim, "cellStarts", counts.Length);
            uint sum = 0;
            for (int i = 0; i < counts.Length; i++) { Require(starts[i] == sum, "Exclusive scan mismatch at " + i); sum += counts[i]; }

            State(sim, new[] { new Vector3(100,10,100), new Vector3(105,10,100) }, new Vector3[2]);
            sim.SimulateStep(1f / 120);
            Require(Read<uint>(sim, "contactCount", 4)[0] == 0, "Separated cubes generated contacts");

            State(sim, new[] { new Vector3(100,10,100), new Vector3(100.95f,10,100) }, new[] { Vector3.right, Vector3.left });
            for (int i = 0; i < 12; i++) sim.SimulateStep(1f / 120);
            var p = Read<Vector3>(sim, "positions", 2);
            var v = Read<Vector3>(sim, "velocities", 2);
            Require(p[1].x - p[0].x > 0.98f, "Overlapping pair did not separate");
            Require(Mathf.Abs(v[0].x + v[1].x) < 0.001f, "Pair lost linear momentum");

            State(sim, new[] { new Vector3(100,10,100), new Vector3(100.95f,10.3f,100.2f) }, new[] { Vector3.right * 2, Vector3.left * 2 });
            sim.SimulateStep(1f / 120);
            var w = Read<Vector3>(sim, "angularVelocities", 2);
            Require(w[0].sqrMagnitude + w[1].sqrMagnitude > 0.001f, "Off-center impact produced no rotation");

            State(sim, new[] { new Vector3(100,0.5f,100), new Vector3(105,0.5f,100) }, new[] { Vector3.right, Vector3.right });
            for (int i = 0; i < 120; i++) sim.SimulateStep(1f / 120);
            v = Read<Vector3>(sim, "velocities", 2);
            Require(Mathf.Abs(v[0].x) < 0.4f, "Floor friction failed to stop sliding");
            CheckBodies(sim, 2);
            Dispose(sim); sim = Create(6, 128);
            State(sim, new[] {new Vector3(0.5f,20,20),new Vector3(199.5f,20,20),new Vector3(20,0.5f,20),new Vector3(20,199.5f,20),new Vector3(20,20,0.5f),new Vector3(20,20,199.5f)},
                new[] {Vector3.left*10,Vector3.right*10,Vector3.down*10,Vector3.up*10,Vector3.back*10,Vector3.forward*10});
            for (int i = 0; i < 8; i++) sim.SimulateStep(1f / 120);
            CheckBodies(sim, 6);
            // Dispatch arguments must span rows when a contact count exceeds 65535 groups.
            var shader = (ComputeShader)Field("simulation").GetValue(sim);
            shader.SetInt("_MaxContacts", 5000000);
            Buffer(sim, "contactCount").SetData(new uint[] {4200000, 0, 0, 0});
            shader.Dispatch(shader.FindKernel("FinalizeContacts"), 1, 1, 1);
            var args = Read<uint>(sim, "dispatchArgs", 3);
            Require(args[0] == 65535 && args[1] == 2 && args[2] == 1, "2D indirect dispatch arguments invalid");
            Dispose(sim); sim = Create(65, 1);
            var positions = new Vector3[65];
            for (int i = 0; i < 65; i++) positions[i] = new Vector3(2 + i * 2, 0.5f, 100);
            State(sim, positions, new Vector3[65]);
            sim.SimulateStep(1f / 120);
            Require(Read<uint>(sim, "contactCount", 4)[0] == 260, "Overflow counter or tail-thread processing failed");
            CheckBodies(sim, 65);
            Call(sim, "Release"); Call(sim, "OnEnable");
            sim.SimulateStep(1f / 120); CheckBodies(sim, 65);
            CheckErrors();
            UnityEngine.Debug.Log("GPU CUBE REGRESSION PASS: million-cell scan, separation, momentum, angular response, friction, six boundaries, 2D dispatch, overflow, 65-body tail, reinitialization.");
        }
        finally { Dispose(sim); Application.logMessageReceived -= CaptureError; }
    }

    [StructLayout(LayoutKind.Sequential)]
    struct ContactData
    {
        public uint a, b;
        public Vector3 point, normal;
        public float penetration, lambda;
        public Vector3 tangent;
        public float target;
    }
    static string ContactQuality(GPUCube_gpt sim)
    {
        int n = (int)Math.Min(Read<uint>(sim, "contactCount", 4)[0], (uint)(int)Field("capacity").GetValue(sim));
        var contacts = Read<ContactData>(sim, "contacts", n);
        float max = 0; double sum = 0;
        foreach (var c in contacts) { max = Mathf.Max(max, c.penetration); sum += Math.Max(0, c.penetration); }
        Require(max < 0.15f && sum / Math.Max(1, n) < 0.03, $"Excessive settled overlap: max={max}, mean={sum / Math.Max(1, n)}");
        ContactData deepest = default;
        foreach (var c in contacts) if (c.penetration >= deepest.penetration) deepest = c;
        var pos = Read<Vector3>(sim, "positions", (int)sim.count);
        var rot = Read<Vector4>(sim, "rotations", (int)sim.count);
        float projection = 0;
        if (deepest.b != uint.MaxValue)
        {
            var ra = rot[deepest.a]; var rb = rot[deepest.b];
            var qa = new Quaternion(ra.x,ra.y,ra.z,ra.w); var qb = new Quaternion(rb.x,rb.y,rb.z,rb.w);
            foreach (var axis in new[] {Vector3.right, Vector3.up, Vector3.forward})
                projection += 0.5f * (Mathf.Abs(Vector3.Dot(qa * axis, deepest.normal)) + Mathf.Abs(Vector3.Dot(qb * axis, deepest.normal)));
            projection -= Mathf.Abs(Vector3.Dot(pos[deepest.a] - pos[deepest.b], deepest.normal));
        }
        return $"max penetration={max:F4}, mean={sum / Math.Max(1, n):F4}, deepest pair={deepest.a}/{deepest.b}, projected overlap={projection:F4}";
    }
    static void CheckBodies(GPUCube_gpt sim, int count)
    {
        var positions = Read<Vector3>(sim, "positions", count);
        var rotations = Read<Vector4>(sim, "rotations", count);
        var velocities = Read<Vector3>(sim, "velocities", count);
        var angular = Read<Vector3>(sim, "angularVelocities", count);
        for (int i = 0; i < count; i++)
        {
            var p = positions[i];
            Require(!float.IsNaN(p.sqrMagnitude) && !float.IsInfinity(p.sqrMagnitude), "Nonfinite position: " + i);
            Require(p.x >= 0.49f && p.y >= 0.49f && p.z >= 0.49f && p.x <= 199.51f && p.y <= 199.51f && p.z <= 199.51f, "Body outside container: " + i);
            Require(Mathf.Abs(rotations[i].sqrMagnitude - 1) < 0.001f, "Quaternion normalization: " + i);
            Require(!float.IsNaN(velocities[i].sqrMagnitude) && !float.IsInfinity(velocities[i].sqrMagnitude) && !float.IsNaN(angular[i].sqrMagnitude) && !float.IsInfinity(angular[i].sqrMagnitude), "Nonfinite velocity: " + i);
            var r = rotations[i]; var q = new Quaternion(r.x,r.y,r.z,r.w);
            Vector3 e = Vector3.zero;
            foreach (var axis in new[] {Vector3.right,Vector3.up,Vector3.forward})
            {
                Vector3 a = q * axis; e += new Vector3(Mathf.Abs(a.x),Mathf.Abs(a.y),Mathf.Abs(a.z)) * 0.5f;
            }
            Require(p.x-e.x >= -0.001f && p.y-e.y >= -0.001f && p.z-e.z >= -0.001f && p.x+e.x <= 200.001f && p.y+e.y <= 200.001f && p.z+e.z <= 200.001f, "Rotated cube outside container: " + i);
        }
        Require(Read<uint>(sim, "contactCount", 4)[3] == 0, "Invalid state fallback triggered");
    }

    static GPUCube_gpt soak;
    static int remaining, total, soakCount;
    static uint peakContacts, peakOccupancy;
    static Stopwatch timer;
    public static string SoakStatus { get; private set; } = "Not started";

    [MenuItem("Tools/GPU Cubes/Run 100k GPU Soak (3 seconds simulated)")]
    public static void Run100k() => StartSoak(100000, 360);
    [MenuItem("Tools/GPU Cubes/Run 200k GPU Soak (3 seconds simulated)")]
    public static void Run200k() => StartSoak(200000, 360);

    public static void StartSoak(int count, int steps)
    {
        Require(!EditorApplication.isPlaying && !soak, "Stop Play mode or wait for the active soak.");
        BeginErrorCapture();
        soak = Create(count, count * 24);
        soakCount = count; total = remaining = steps; peakContacts = peakOccupancy = 0;
        timer = Stopwatch.StartNew(); SoakStatus = "Running";
        EditorApplication.update += SoakTick;
    }
    static void SoakTick()
    {
        try
        {
            // One submitted step per editor tick; read back four counters to fence the GPU.
            soak.SimulateStep(1f / 120);
            var stats = Read<uint>(soak, "contactCount", 4);
            CheckErrors();
            peakContacts = Math.Max(peakContacts, stats[0]); peakOccupancy = Math.Max(peakOccupancy, stats[1]);
            Require(stats[0] <= soakCount * 24, "Contact capacity exceeded");
            Require(stats[3] == 0, "Nonfinite body state");
            remaining--;
            SoakStatus = $"Running {total - remaining}/{total}; peak contacts={peakContacts}; wall={timer.Elapsed.TotalSeconds:F1}s";
            if (remaining > 0) return;
            CheckBodies(soak, soakCount);
            Require(peakContacts > 0, "No contacts during soak");
            SoakStatus = $"PASS: {soakCount:N0} cubes, {total} steps, {total / 120f:F2}s simulated, {timer.Elapsed.TotalSeconds:F2}s wall (includes Editor ticks/readbacks), peak contacts={peakContacts:N0}, peak cell={peakOccupancy}, {ContactQuality(soak)}, GPU={SystemInfo.graphicsDeviceName}, buffers={soak.AllocatedBytes / 1048576f:F1} MiB";
            UnityEngine.Debug.Log(SoakStatus);
            StopSoak();
        }
        catch (Exception e)
        {
            SoakStatus = "FAIL: " + e;
            UnityEngine.Debug.LogError(SoakStatus);
            StopSoak();
        }
    }
    [MenuItem("Tools/GPU Cubes/Stop GPU Soak")]
    public static void StopSoak()
    {
        EditorApplication.update -= SoakTick;
        Application.logMessageReceived -= CaptureError;
        Dispose(soak); soak = null;
    }
}
