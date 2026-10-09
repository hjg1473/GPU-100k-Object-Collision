using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

/// <summary>GPU-only, unit-mass, unit-size oriented boxes. No per-cube GameObjects/PhysX bodies.</summary>
public class GPUCube_gpt : MonoBehaviour
{
    public ComputeShader computeShader;
    public Material mat;
    public Mesh mesh;
    public uint count = 100000;
    [SerializeField, Min(1)] int maxContacts = 2400000;
    [SerializeField, Range(1, 64)] int solverIterations = 24;
    [SerializeField, Range(30, 240)] int simulationHz = 120;
    [SerializeField, Range(1, 8)] int maxSubsteps = 4;
    [SerializeField, Range(0, 1)] float friction = 0.6f;
    [SerializeField, Range(0, 1)] float restitution = 0.1f;
    [SerializeField, Min(0)] float linearDamping = 0.05f;
    [SerializeField, Min(0)] float angularDamping = 0.1f;
    [SerializeField, Range(1, 64)] int spawnLayers = 8;
    [SerializeField] bool diagnostics = true;

    // These are sampled asynchronously, once per second; they describe one substep.
    public uint RequestedContacts { get; private set; }
    public uint PeakCellOccupancy { get; private set; }
    public uint InvalidBodies { get; private set; }
    public bool ContactOverflow => RequestedContacts > capacity;
    public long AllocatedBytes { get; private set; }
    public double SimulatedSeconds { get; private set; }
    public float DroppedSimulationSeconds { get; private set; }

    const int GridSize = 100;
    const float CellSize = 2f; // >= unit cube's circumscribed diameter sqrt(3).
    const int Cells = GridSize * GridSize * GridSize;
    const int Threads = 64;
    const int ScanThreads = 256;
    const float WorldSize = GridSize * CellSize;
    readonly List<GraphicsBuffer> buffers = new List<GraphicsBuffer>();
    readonly List<GraphicsBuffer> scanSums = new List<GraphicsBuffer>();
    readonly List<GraphicsBuffer> scanOffsets = new List<GraphicsBuffer>();
    readonly List<int> scanLengths = new List<int>();
    ComputeShader simulation;
    Material renderingMaterial;
    GraphicsBuffer positions, rotations, velocities, nextVelocities, angularVelocities, nextAngularVelocities;
    GraphicsBuffer cellCounts, cellStarts, cellWrites, objectIDs;
    GraphicsBuffer contacts, contactCount, deltas, heads, degrees, nextA, nextB, dispatchArgs, drawArgs;
    int init, clearGrid, countCells, scatter, scan, addOffsets, resetCounters;
    int gravity, boundaryContacts, buildContacts, clearHeads, linkContacts, finalizeContacts, solve, apply, integrate;
    int bodyCount, capacity, generation;
    float accumulator, nextDiagnostics;
    bool initialized, readbackPending, overflowReported;

    GraphicsBuffer Allocate(int length, int stride, GraphicsBuffer.Target target = GraphicsBuffer.Target.Structured)
    {
        long bytes = (long)length * stride;
        if (bytes > SystemInfo.maxGraphicsBufferSize)
            throw new InvalidOperationException($"GPU buffer requires {bytes:N0} bytes, above this device's limit.");
        var buffer = new GraphicsBuffer(target, length, stride);
        buffers.Add(buffer);
        AllocatedBytes += bytes;
        return buffer;
    }

    void OnEnable()
    {
        try
        {
            if (!computeShader || !mat || !mesh)
                throw new InvalidOperationException("Assign 7_Compute, 7_Cube material and a centered unit cube mesh.");
            if (!SystemInfo.supportsComputeShaders || !SystemInfo.supportsInstancing)
                throw new NotSupportedException("Compute shaders and indirect instancing are required.");
            if (count == 0 || count > 1000000 || maxContacts < 1 || maxContacts > 32000000)
                throw new ArgumentOutOfRangeException("Use 1..1,000,000 bodies and 1..32,000,000 contacts.");
            if ((mesh.bounds.size - Vector3.one).sqrMagnitude > 0.0001f || mesh.bounds.center.sqrMagnitude > 0.0001f)
                throw new InvalidOperationException("Physics uses centered 1 x 1 x 1 cubes; the mesh bounds must match.");
            bodyCount = (int)count;
            capacity = maxContacts;
            simulation = Instantiate(computeShader); // Independent bindings for multiple simulators.
            renderingMaterial = new Material(mat) { enableInstancing = true };
            init = Kernel("InitializeBodies"); clearGrid = Kernel("ClearGrid");
            countCells = Kernel("UpdateCellCounts"); scatter = Kernel("UpdateObjectIDs");
            scan = Kernel("ScanBlocks"); addOffsets = Kernel("AddBlockOffsets");
            resetCounters = Kernel("ResetCounters"); gravity = Kernel("ApplyGravity");
            boundaryContacts = Kernel("BuildFloorContacts"); buildContacts = Kernel("BuildContacts");
            clearHeads = Kernel("ClearContactHeads"); linkContacts = Kernel("BuildContactLinks");
            finalizeContacts = Kernel("FinalizeContacts"); solve = Kernel("SolveContactImpulses");
            apply = Kernel("ApplyContactImpulses"); integrate = Kernel("main");

            positions = Allocate(bodyCount, 12); rotations = Allocate(bodyCount, 16);
            velocities = Allocate(bodyCount, 12); nextVelocities = Allocate(bodyCount, 12);
            angularVelocities = Allocate(bodyCount, 12); nextAngularVelocities = Allocate(bodyCount, 12);
            cellCounts = Allocate(Cells, 4); cellStarts = Allocate(Cells, 4); cellWrites = Allocate(Cells, 4);
            objectIDs = Allocate(bodyCount, 4);
            // HLSL ContactConstraint: 2 uint + 2 float3 + 2 float + float3 + float = 56 bytes.
            contacts = Allocate(capacity, 56); deltas = Allocate(capacity, 12);
            nextA = Allocate(capacity, 4); nextB = Allocate(capacity, 4);
            heads = Allocate(bodyCount, 4); degrees = Allocate(bodyCount, 4);
            contactCount = Allocate(4, 4);
            dispatchArgs = Allocate(3, 4, GraphicsBuffer.Target.IndirectArguments | GraphicsBuffer.Target.Raw);
            drawArgs = Allocate(1, GraphicsBuffer.IndirectDrawIndexedArgs.size, GraphicsBuffer.Target.IndirectArguments);
            drawArgs.SetData(new[] { new GraphicsBuffer.IndirectDrawIndexedArgs {
                indexCountPerInstance = mesh.GetIndexCount(0), instanceCount = (uint)bodyCount,
                startIndex = mesh.GetIndexStart(0), baseVertexIndex = (uint)mesh.GetBaseVertex(0)
            }});
            for (int length = Cells; ; length = DivideUp(length, ScanThreads))
            {
                int blocks = DivideUp(length, ScanThreads);
                scanLengths.Add(length);
                scanSums.Add(Allocate(blocks, 4));
                scanOffsets.Add(Allocate(blocks, 4));
                if (blocks == 1) break;
            }
            simulation.SetInt("_Count", bodyCount);
            simulation.SetInt("_MaxContacts", capacity);
            simulation.SetInt("_GridSize", GridSize);
            simulation.SetFloat("_CellSize", CellSize);
            simulation.SetVector("_WorldMin", Vector3.zero);
            int width = Mathf.Clamp(Mathf.CeilToInt(Mathf.Sqrt(bodyCount / (float)Mathf.Max(1, spawnLayers))), 1, 170);
            simulation.SetInt("_SpawnWidth", width);
            BindStaticBuffers();
            simulation.Dispatch(resetCounters, 1, 1, 1);
            DispatchBodies(init);
            renderingMaterial.SetBuffer("_Positions", positions);
            renderingMaterial.SetBuffer("_Rotations", rotations);
            initialized = true;
            Debug.Log($"GPU cubes: {bodyCount:N0} bodies, {capacity:N0} contact capacity, {AllocatedBytes / 1048576f:F1} MiB buffers.", this);
        }
        catch (Exception e)
        {
            Debug.LogException(e, this);
            Release();
            enabled = false;
        }
    }

    int Kernel(string name) => simulation.FindKernel(name);
    static int DivideUp(int n, int divisor) => (n + divisor - 1) / divisor;
    void Bind(int kernel, string name, GraphicsBuffer buffer) => simulation.SetBuffer(kernel, name, buffer);
    void DispatchBodies(int kernel) => simulation.Dispatch(kernel, DivideUp(bodyCount, Threads), 1, 1);

    void BindStaticBuffers()
    {
        foreach (int k in new[] { init, countCells, scatter, gravity, boundaryContacts, buildContacts, solve, apply, integrate })
            Bind(k, "_Positions", positions);
        foreach (int k in new[] { init, boundaryContacts, buildContacts, integrate }) Bind(k, "_Rotations", rotations);
        Bind(init, "_NextVelocities", velocities); Bind(init, "_NextAngularVelocities", angularVelocities);
        foreach (int k in new[] { clearGrid, countCells, buildContacts }) Bind(k, "_CellCounts", cellCounts);
        foreach (int k in new[] { clearGrid, scatter }) Bind(k, "_CellWriteCounters", cellWrites);
        foreach (int k in new[] { scatter, buildContacts })
        {
            Bind(k, "_CellStarts", cellStarts); Bind(k, "_ObjectIDs", objectIDs);
        }
        foreach (int k in new[] { boundaryContacts, buildContacts, linkContacts, solve, apply })
            Bind(k, "_ContactConstraints", contacts);
        foreach (int k in new[] { resetCounters, countCells, boundaryContacts, buildContacts, linkContacts, finalizeContacts, solve, integrate })
            Bind(k, "_ContactCount", contactCount);
        foreach (int k in new[] { clearHeads, linkContacts, apply }) Bind(k, "_BodyContactHeads", heads);
        foreach (int k in new[] { clearHeads, linkContacts, solve }) Bind(k, "_BodyContactCounts", degrees);
        foreach (int k in new[] { linkContacts, apply })
        {
            Bind(k, "_ContactNextA", nextA); Bind(k, "_ContactNextB", nextB);
        }
        Bind(integrate, "_ReadContacts", contacts); Bind(integrate, "_ReadHeads", heads);
        Bind(integrate, "_ReadNextA", nextA); Bind(integrate, "_ReadNextB", nextB);
        Bind(finalizeContacts, "_DispatchArgs", dispatchArgs);
        Bind(solve, "_ContactDeltaImpulses", deltas); Bind(apply, "_ContactDeltaImpulses", deltas);
    }

    void Update()
    {
        if (!initialized) return;
        float step = 1f / Mathf.Clamp(simulationHz, 30, 240);
        float elapsed = Mathf.Max(0, Time.deltaTime);
        float budget = step * Mathf.Clamp(maxSubsteps, 1, 8);
        DroppedSimulationSeconds += Mathf.Max(0, accumulator + elapsed - budget);
        accumulator = Mathf.Min(accumulator + elapsed, budget);
        while (accumulator + 1e-7f >= step)
        {
            SimulateStep(step);
            accumulator = Mathf.Max(0, accumulator - step);
        }
        var rp = new RenderParams(renderingMaterial) {
            worldBounds = new Bounds(Vector3.one * WorldSize * 0.5f, Vector3.one * WorldSize),
            shadowCastingMode = ShadowCastingMode.Off, receiveShadows = false
        };
        Graphics.RenderMeshIndirect(rp, mesh, drawArgs);
        if (diagnostics && Time.unscaledTime >= nextDiagnostics) RequestDiagnostics();
    }

    // Explicit stepping is also used by the Editor GPU validation harness.
    public void SimulateStep(float dt)
    {
        if (!initialized) throw new InvalidOperationException("Simulation is not initialized.");
        if (!(dt > 0 && dt <= 1f / 30f)) throw new ArgumentOutOfRangeException(nameof(dt));
        simulation.SetFloat("_DeltaTime", dt);
        simulation.SetFloat("_Friction", Mathf.Clamp01(friction));
        simulation.SetFloat("_Restitution", Mathf.Clamp01(restitution));
        simulation.SetFloat("_LinearDamping", Mathf.Max(0, linearDamping));
        simulation.SetFloat("_AngularDamping", Mathf.Max(0, angularDamping));
        simulation.Dispatch(resetCounters, 1, 1, 1);
        simulation.Dispatch(clearGrid, DivideUp(Cells, Threads), 1, 1);
        DispatchBodies(countCells);
        BuildCellStarts();
        DispatchBodies(scatter);
        Bind(gravity, "_Velocities", velocities); Bind(gravity, "_NextVelocities", nextVelocities);
        DispatchBodies(gravity);
        (velocities, nextVelocities) = (nextVelocities, velocities);
        DispatchBodies(clearHeads);
        DispatchBodies(boundaryContacts); // Boundary contacts get capacity before body pairs.
        DispatchBodies(buildContacts);
        simulation.Dispatch(finalizeContacts, 1, 1, 1);
        simulation.DispatchIndirect(linkContacts, dispatchArgs);
        for (int i = 0; i < Mathf.Clamp(solverIterations, 1, 64); i++)
        {
            simulation.SetInt("_SolverIteration", i);
            Bind(solve, "_Velocities", velocities); Bind(solve, "_AngularVelocities", angularVelocities);
            Bind(apply, "_Velocities", velocities); Bind(apply, "_AngularVelocities", angularVelocities);
            Bind(apply, "_NextVelocities", nextVelocities);
            Bind(apply, "_NextAngularVelocities", nextAngularVelocities);
            simulation.DispatchIndirect(solve, dispatchArgs);
            DispatchBodies(apply);
            (velocities, nextVelocities) = (nextVelocities, velocities);
            (angularVelocities, nextAngularVelocities) = (nextAngularVelocities, angularVelocities);
        }
        Bind(integrate, "_Velocities", velocities); Bind(integrate, "_AngularVelocities", angularVelocities);
        Bind(integrate, "_NextVelocities", nextVelocities); Bind(integrate, "_NextAngularVelocities", nextAngularVelocities);
        DispatchBodies(integrate);
        (velocities, nextVelocities) = (nextVelocities, velocities);
        (angularVelocities, nextAngularVelocities) = (nextAngularVelocities, angularVelocities);
        SimulatedSeconds += dt;
    }

    void BuildCellStarts()
    {
        // Three upward block scans and two downward adds for 1,000,000 cells.
        for (int i = 0; i < scanLengths.Count; i++)
        {
            simulation.SetInt("_ScanLength", scanLengths[i]);
            Bind(scan, "_PrefixInput", i == 0 ? cellCounts : scanSums[i - 1]);
            Bind(scan, "_PrefixOutput", i == 0 ? cellStarts : scanOffsets[i - 1]);
            Bind(scan, "_BlockSums", scanSums[i]);
            simulation.Dispatch(scan, DivideUp(scanLengths[i], ScanThreads), 1, 1);
        }
        for (int i = scanLengths.Count - 2; i >= 0; i--)
        {
            simulation.SetInt("_ScanLength", scanLengths[i]);
            Bind(addOffsets, "_PrefixOutput", i == 0 ? cellStarts : scanOffsets[i - 1]);
            Bind(addOffsets, "_BlockOffsets", scanOffsets[i]);
            simulation.Dispatch(addOffsets, DivideUp(scanLengths[i], ScanThreads), 1, 1);
        }
    }

    void RequestDiagnostics()
    {
        if (readbackPending || !SystemInfo.supportsAsyncGPUReadback) return;
        nextDiagnostics = Time.unscaledTime + 1f;
        readbackPending = true;
        int requestGeneration = generation;
        AsyncGPUReadback.Request(contactCount, request => {
            if (this == null || generation != requestGeneration) return;
            readbackPending = false;
            if (request.hasError || !initialized) return;
            var data = request.GetData<uint>();
            RequestedContacts = data[0]; PeakCellOccupancy = data[1]; InvalidBodies = data[3];
            if (ContactOverflow && !overflowReported)
                Debug.LogWarning($"GPU contact overflow: {RequestedContacts:N0} / {capacity:N0}. Contacts were dropped; increase Max Contacts and restart.", this);
            overflowReported = ContactOverflow;
            if (InvalidBodies > 0) Debug.LogError($"GPU solver produced {InvalidBodies} invalid bodies.", this);
        });
    }

    static void DestroyOwned(UnityEngine.Object value)
    {
        if (!value) return;
        if (Application.isPlaying) Destroy(value); else DestroyImmediate(value);
    }

    void OnDisable() => Release();
    void OnDestroy() => Release();
    void Release()
    {
        initialized = false;
        generation++;
        readbackPending = false;
        foreach (var buffer in buffers) buffer.Release();
        buffers.Clear(); scanSums.Clear(); scanOffsets.Clear(); scanLengths.Clear();
        DestroyOwned(simulation);
        DestroyOwned(renderingMaterial);
        simulation = null; renderingMaterial = null;
        AllocatedBytes = 0; accumulator = 0; nextDiagnostics = 0;
        SimulatedSeconds = 0; DroppedSimulationSeconds = 0;
        RequestedContacts = 0; PeakCellOccupancy = 0; InvalidBodies = 0; overflowReported = false;
    }
}
