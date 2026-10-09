using UnityEngine;

public class GPUCube_rigidbody : MonoBehaviour
{

    public ComputeShader computeShader;
    public Material mat;
    public Mesh mesh;
    public uint count = 1000000;
    [SerializeField] int maxContacts = 4000000;
    [SerializeField] int solverIterations = 4;

    
    int mainKernel;
    int clearCellCountsKernel;    
    int clearCellWriteCountersKernel;    
    int updateCellCountsKernel;
    int updateObjectIDsKernel;
    int updateCellStartsKernel;
    int inclusiveToExclusiveKernel;
    int applyGravityKernel;
    int clearContactHeadsKernel;
    int buildContactLinksKernel;
    int solveContactImpulsesKernel;
    int applyContactImpulsesKernel;
    int buildContactsKernel;
    int buildFloorContactsKernel;

    GraphicsBuffer commandBuffer;
    GraphicsBuffer.IndirectDrawIndexedArgs[] commandData;
    GraphicsBuffer positionBuffer;
    GraphicsBuffer collisionBuffer;
    
    GraphicsBuffer cellCountBuffer; 
    GraphicsBuffer objectIDBuffer;
    GraphicsBuffer cellStartBuffer;
    GraphicsBuffer cellWriteCounterBuffer;
    GraphicsBuffer prefixInputBuffer;
    GraphicsBuffer prefixOutputBuffer;

    GraphicsBuffer velocityBuffer;
    GraphicsBuffer newVelocityBuffer;

    GraphicsBuffer rotationBuffer;
    GraphicsBuffer angularVelocityBuffer;
    GraphicsBuffer newAngularVelocityBuffer;

    
    GraphicsBuffer contactConstraintsBuffer;
    GraphicsBuffer contactCountBuffer;
    GraphicsBuffer contactDeltaBuffer;
    GraphicsBuffer bodyContactHeadsBuffer;
    GraphicsBuffer contactNextABuffer;
    GraphicsBuffer contactNextBBuffer;
    

    // const int WorldMin = 25;
    const int commandCount = 1;
    const int GridSize = 100;
    const float CellSize = 2;
    const int CellRange = GridSize * GridSize * GridSize;
    readonly uint[] zeroContactCount = { 0 };


    void Start()
    {
        CreateBuffer();
        

        FindKernel();

        InitPositions();
        InitRotations();
        InitVelocities();
        InitAngularVelocities();

        BindBuffer();
        BindUniformGridBuffer();
        BindContactBuffers();
    }

    void CreateBuffer()
    {
        commandBuffer = new GraphicsBuffer(GraphicsBuffer.Target.IndirectArguments, commandCount, GraphicsBuffer.IndirectDrawIndexedArgs.size);
        commandData = new GraphicsBuffer.IndirectDrawIndexedArgs[commandCount];
        commandData[0].indexCountPerInstance = mesh.GetIndexCount(0);
        commandData[0].instanceCount = count;
        commandBuffer.SetData(commandData); 

        positionBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, (int)count, sizeof(float) * 3);
        velocityBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, (int)count, sizeof(float) * 3);
        newVelocityBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, (int)count, sizeof(float) * 3);

        rotationBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, (int)count, sizeof(float) * 4);
        angularVelocityBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, (int)count, sizeof(float) * 3);
        newAngularVelocityBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, (int)count, sizeof(float) * 3);

        collisionBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, (int)count, sizeof(uint));

        cellCountBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, CellRange, sizeof(uint));
        objectIDBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, (int)count, sizeof(uint));
        cellStartBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, CellRange, sizeof(uint));
        cellWriteCounterBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, CellRange, sizeof(uint));
        prefixInputBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, CellRange, sizeof(uint));
        prefixOutputBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, CellRange, sizeof(uint));

        contactConstraintsBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, maxContacts, 40);
        contactCountBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 1, sizeof(uint));
        contactDeltaBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, maxContacts, sizeof(float));
        bodyContactHeadsBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, (int)count, sizeof(uint));
        contactNextABuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, maxContacts, sizeof(uint));
        contactNextBBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, maxContacts, sizeof(uint));
    }

    void FindKernel()
    {
        mainKernel = computeShader.FindKernel("main");

        clearCellCountsKernel = computeShader.FindKernel("ClearCellCounts");
        clearCellWriteCountersKernel = computeShader.FindKernel("ClearCellWriteCounters");

        updateCellCountsKernel = computeShader.FindKernel("UpdateCellCounts");
        updateObjectIDsKernel = computeShader.FindKernel("UpdateObjectIDs");
        updateCellStartsKernel = computeShader.FindKernel("UpdateCellStarts");
        inclusiveToExclusiveKernel = computeShader.FindKernel("InclusiveToExclusive");

        applyGravityKernel = computeShader.FindKernel("ApplyGravity");
        
        clearContactHeadsKernel = computeShader.FindKernel("ClearContactHeads");
        buildContactLinksKernel = computeShader.FindKernel("BuildContactLinks");
        solveContactImpulsesKernel = computeShader.FindKernel("SolveContactImpulses");
        applyContactImpulsesKernel = computeShader.FindKernel("ApplyContactImpulses");
        buildContactsKernel = computeShader.FindKernel("BuildContacts");
        buildFloorContactsKernel = computeShader.FindKernel("BuildFloorContacts");
    }

    void BindBuffer()
    {
        computeShader.SetInt("_Count", (int)count);
        
        computeShader.SetInt("_GridSize", GridSize);
        computeShader.SetFloat("_CellSize", CellSize);

        computeShader.SetVector("_WorldMin", new Vector3(0,0,0));

        computeShader.SetBuffer(mainKernel, "_Positions", positionBuffer);
        computeShader.SetBuffer(mainKernel, "_Rotations", rotationBuffer);

        computeShader.SetBuffer(applyGravityKernel, "_Positions", positionBuffer);
        // computeShader.SetBuffer(mainKernel, "_Collisions", collisionBuffer);
        // computeShader.SetBuffer(mainKernel, "_CellCounts", cellCountBuffer);
        // computeShader.SetBuffer(mainKernel, "_CellStarts", cellStartBuffer);
        // computeShader.SetBuffer(mainKernel, "_ObjectIDs", objectIDBuffer);

        mat.SetBuffer("_Positions", positionBuffer);
        mat.SetBuffer("_Rotations", rotationBuffer);
        mat.SetBuffer("_Collisions", collisionBuffer);
    }

    void BindUniformGridBuffer()
    {
        computeShader.SetBuffer(clearCellCountsKernel, "_CellCounts", cellCountBuffer);
        computeShader.SetBuffer(clearCellWriteCountersKernel, "_CellWriteCounters", cellWriteCounterBuffer);

        computeShader.SetBuffer(updateCellCountsKernel, "_Positions", positionBuffer);
        computeShader.SetBuffer(updateCellCountsKernel, "_CellCounts", cellCountBuffer);

        computeShader.SetBuffer(updateObjectIDsKernel, "_Positions", positionBuffer);
        computeShader.SetBuffer(updateObjectIDsKernel, "_CellCounts", cellCountBuffer);
        computeShader.SetBuffer(updateObjectIDsKernel, "_ObjectIDs", objectIDBuffer);
        computeShader.SetBuffer(updateObjectIDsKernel, "_CellStarts", cellStartBuffer);
        computeShader.SetBuffer(updateObjectIDsKernel, "_CellWriteCounters", cellWriteCounterBuffer);
    }

    void BindContactBuffers()
    {
        // ClearContactHeads
        computeShader.SetBuffer(clearContactHeadsKernel, "_BodyContactHeads", bodyContactHeadsBuffer);

        // BuildFloorContacts
        computeShader.SetBuffer(buildFloorContactsKernel, "_Positions", positionBuffer);
        computeShader.SetBuffer(buildFloorContactsKernel, "_Rotations", rotationBuffer);
        computeShader.SetBuffer(buildFloorContactsKernel, "_ContactConstraints", contactConstraintsBuffer);
        computeShader.SetBuffer(buildFloorContactsKernel, "_ContactCount", contactCountBuffer);

        // BuildContacts
        computeShader.SetBuffer(buildContactsKernel, "_Positions", positionBuffer);
        computeShader.SetBuffer(buildContactsKernel, "_Rotations", rotationBuffer);
        computeShader.SetBuffer(buildContactsKernel, "_CellCounts", cellCountBuffer);
        computeShader.SetBuffer(buildContactsKernel, "_CellStarts", cellStartBuffer);
        computeShader.SetBuffer(buildContactsKernel, "_ObjectIDs", objectIDBuffer);
        computeShader.SetBuffer(buildContactsKernel, "_ContactConstraints", contactConstraintsBuffer);
        computeShader.SetBuffer(buildContactsKernel, "_ContactCount", contactCountBuffer);

        // BuildContactLinks
        computeShader.SetBuffer(buildContactLinksKernel, "_ContactConstraints", contactConstraintsBuffer);
        computeShader.SetBuffer(buildContactLinksKernel, "_ContactCount", contactCountBuffer);
        computeShader.SetBuffer(buildContactLinksKernel, "_BodyContactHeads", bodyContactHeadsBuffer);
        computeShader.SetBuffer(buildContactLinksKernel, "_ContactNextA", contactNextABuffer);
        computeShader.SetBuffer(buildContactLinksKernel, "_ContactNextB", contactNextBBuffer);

        // SolveContactImpulses
        computeShader.SetBuffer(solveContactImpulsesKernel, "_ContactConstraints", contactConstraintsBuffer);
        computeShader.SetBuffer(solveContactImpulsesKernel, "_ContactCount", contactCountBuffer);
        computeShader.SetBuffer(solveContactImpulsesKernel, "_ContactDeltaImpulses", contactDeltaBuffer);
        computeShader.SetBuffer(solveContactImpulsesKernel, "_Positions", positionBuffer);

        // ApplyContactImpulses
        computeShader.SetBuffer(applyContactImpulsesKernel, "_ContactConstraints", contactConstraintsBuffer);
        computeShader.SetBuffer(applyContactImpulsesKernel, "_ContactDeltaImpulses", contactDeltaBuffer);
        computeShader.SetBuffer(applyContactImpulsesKernel, "_BodyContactHeads", bodyContactHeadsBuffer);
        computeShader.SetBuffer(applyContactImpulsesKernel, "_ContactNextA", contactNextABuffer);
        computeShader.SetBuffer(applyContactImpulsesKernel, "_ContactNextB", contactNextBBuffer);
        computeShader.SetBuffer(applyContactImpulsesKernel, "_Positions", positionBuffer);

        computeShader.SetInt("_MaxContacts", maxContacts);
    }

    void OnDestroy()
    {
        commandBuffer?.Release();
        commandBuffer = null;

        positionBuffer?.Release();
        positionBuffer = null;

        collisionBuffer?.Release();
        collisionBuffer = null;
        
        cellCountBuffer?.Release();
        cellCountBuffer = null;
        
        objectIDBuffer?.Release();
        objectIDBuffer = null;
        
        cellStartBuffer?.Release();
        cellStartBuffer = null;
        
        cellWriteCounterBuffer?.Release();
        cellWriteCounterBuffer = null;

        prefixInputBuffer?.Release();
        prefixInputBuffer = null;

        prefixOutputBuffer?.Release();
        prefixOutputBuffer = null;

        velocityBuffer?.Release();
        velocityBuffer = null;

        newVelocityBuffer?.Release();
        newVelocityBuffer = null;
        
        rotationBuffer?.Release();
        rotationBuffer = null;

        angularVelocityBuffer?.Release();
        angularVelocityBuffer = null;

        newAngularVelocityBuffer?.Release();
        newAngularVelocityBuffer = null;

        
        contactConstraintsBuffer?.Release();
        contactConstraintsBuffer = null;
        
        contactCountBuffer?.Release();
        contactCountBuffer = null;
        
        contactDeltaBuffer?.Release();
        contactDeltaBuffer = null;
        
        bodyContactHeadsBuffer?.Release();
        bodyContactHeadsBuffer = null;

        contactNextABuffer?.Release();
        contactNextABuffer = null;

        contactNextBBuffer?.Release();
        contactNextBBuffer = null;
    }

    void Update()
    {
        UpdateSimulationParameters();

        BuildUniformGrid();
        ApplyGravity();
        SolveContacts();
        IntegrateBodies();
        RenderObjects();
    }
    void UpdateSimulationParameters()
    {
        computeShader.SetFloat("_DeltaTime", Time.deltaTime);
    }
    
    void BuildUniformGrid()
    {
        computeShader.Dispatch(clearCellCountsKernel, Mathf.CeilToInt(CellRange / 64.0f), 1, 1);
        computeShader.Dispatch(clearCellWriteCountersKernel, Mathf.CeilToInt(CellRange / 64.0f), 1, 1);
        computeShader.Dispatch(updateCellCountsKernel, Mathf.CeilToInt(count / 64.0f), 1, 1);
        
        Hillis_Steele();

        computeShader.Dispatch(updateObjectIDsKernel, Mathf.CeilToInt(count / 64.0f), 1, 1);

    }

    void SolveContacts()
    {
        BuildContactConstraints();
        BuildContactLinks();
        RunJacobiIterations();
    }

    void IntegrateBodies()
    {
        computeShader.SetBuffer(mainKernel, "_Velocities", velocityBuffer);
        computeShader.SetBuffer(mainKernel, "_AngularVelocities", angularVelocityBuffer);
        computeShader.Dispatch(mainKernel, Mathf.CeilToInt(count / 64.0f), 1, 1);
    }

    void RenderObjects()
    {
        RenderParams rp = new RenderParams(mat);
        rp.worldBounds = new Bounds(Vector3.zero, 10000*Vector3.one);        
        Graphics.RenderMeshIndirect(rp, mesh, commandBuffer, commandCount);
    }

    void Hillis_Steele()
    {
        GraphicsBuffer input;
        GraphicsBuffer output;

        computeShader.SetInt("_Offset", 1);

        computeShader.SetBuffer(updateCellStartsKernel, "_PrefixInput", cellCountBuffer);
        computeShader.SetBuffer(updateCellStartsKernel, "_PrefixOutput", prefixInputBuffer);

        computeShader.Dispatch(updateCellStartsKernel, Mathf.CeilToInt(CellRange / 64.0f), 1, 1);

        input = prefixInputBuffer;
        output = prefixOutputBuffer;

        for (int offset = 2; offset < CellRange; offset *= 2)
        {
            computeShader.SetInt("_Offset", offset);

            computeShader.SetBuffer(updateCellStartsKernel, "_PrefixInput", input);
            computeShader.SetBuffer(updateCellStartsKernel, "_PrefixOutput", output);

            computeShader.Dispatch(updateCellStartsKernel, Mathf.CeilToInt(CellRange / 64.0f), 1, 1);

            (input, output) = (output, input);
        }
        // exclusive 변환 (right shift >>)
        computeShader.SetBuffer(inclusiveToExclusiveKernel, "_PrefixInput", input);
        computeShader.SetBuffer(inclusiveToExclusiveKernel, "_CellStarts", cellStartBuffer);
        computeShader.Dispatch(inclusiveToExclusiveKernel, Mathf.CeilToInt(CellRange / 64.0f), 1, 1);
    }

    void ApplyGravity()
    {
        computeShader.SetBuffer(applyGravityKernel, "_Velocities", velocityBuffer);
        computeShader.SetBuffer(applyGravityKernel, "_NextVelocities", newVelocityBuffer);

        computeShader.Dispatch(applyGravityKernel, Mathf.CeilToInt(count / 64.0f), 1, 1);

        (velocityBuffer, newVelocityBuffer) = (newVelocityBuffer, velocityBuffer);
    }

    void InitPositions()
    {
        Vector3[] positions = new Vector3[count];

        for (int i=0; i < count; i++)
        {
            int x = i % 100;
            int y = (i / 100) % 100;
            int z = i / 10000;

            // positions[i] = new Vector3(x * 1.2f, y * 1.2f, z * 1.2f);
            positions[i] = new Vector3(10f, i * 5f + 1, 10f);
        }
        
        positionBuffer.SetData(positions);
    }

    void InitVelocities()
    {
        Vector3[] velocities = new Vector3[count];

        for (int i=0; i < count; i++)
        {
            // velocities[i] = new Vector3(Random.Range(-1,1), Random.Range(-1,1), Random.Range(-1,1));
            velocities[i] = new Vector3(0, 0, 0);
        }
        velocityBuffer.SetData(velocities);
        newVelocityBuffer.SetData(velocities);
    }

    void InitAngularVelocities()
    {
        Vector3[] angularVelocities = new Vector3[count];

        angularVelocityBuffer.SetData(angularVelocities);
        newAngularVelocityBuffer.SetData(angularVelocities);
    }

    void InitRotations()
    {
        Vector4[] rotations = new Vector4[count];

        for (int i = 0; i < count; i++)
        {
            rotations[i] = new Vector4(0, 0, 0, 1);
        }
        // rotations[0] = new Vector4(0, 0, 0, 1);
        // rotations[1] = new Vector4(0, 0, 0, 1);

        rotationBuffer.SetData(rotations);
    }

    void BuildContactConstraints()
    {
        contactCountBuffer.SetData(zeroContactCount);

        computeShader.Dispatch(clearContactHeadsKernel, Mathf.CeilToInt(count / 64.0f), 1, 1);
        computeShader.Dispatch(buildContactsKernel, Mathf.CeilToInt(count / 64.0f), 1, 1);
        computeShader.Dispatch(buildFloorContactsKernel, Mathf.CeilToInt(count / 64.0f), 1, 1);
    }

    void BuildContactLinks()
    {
        computeShader.Dispatch(buildContactLinksKernel, (maxContacts + 63) / 64, 1, 1);
    }

    void BindSolverVelocities()
    {
        computeShader.SetBuffer(solveContactImpulsesKernel, "_Velocities", velocityBuffer);
        computeShader.SetBuffer(solveContactImpulsesKernel, "_AngularVelocities", angularVelocityBuffer);
        computeShader.SetBuffer(applyContactImpulsesKernel, "_Velocities", velocityBuffer);
        computeShader.SetBuffer(applyContactImpulsesKernel, "_NextVelocities", newVelocityBuffer);
        computeShader.SetBuffer(applyContactImpulsesKernel, "_AngularVelocities", angularVelocityBuffer);
        computeShader.SetBuffer(applyContactImpulsesKernel, "_NextAngularVelocities", newAngularVelocityBuffer);
    }

    void SwapVelocityBuffers()
    {
        (velocityBuffer, newVelocityBuffer) = (newVelocityBuffer, velocityBuffer);
        (angularVelocityBuffer, newAngularVelocityBuffer) = (newAngularVelocityBuffer, angularVelocityBuffer);
    }
    void RunJacobiIterations()
    {
        for (int i = 0; i < solverIterations; i++)
        {
            BindSolverVelocities();

            computeShader.Dispatch(solveContactImpulsesKernel, (maxContacts + 63) / 64, 1, 1);
            computeShader.Dispatch(applyContactImpulsesKernel, Mathf.CeilToInt(count / 64.0f), 1, 1);

            SwapVelocityBuffers();
        }
    }

    void DebugGrid()
    {
        uint[] cellCounts = new uint[CellRange];
        cellCountBuffer.GetData(cellCounts);

        int occupiedCells = 0;
        int totalObjects = 0;

        foreach (uint cellCount in cellCounts)
        {
            if (cellCount > 0)
                occupiedCells++;

            totalObjects += (int)cellCount;
        }

        Debug.Log(
            $"Grid Objects: {totalObjects}, " +
            $"Occupied Cells: {occupiedCells}"
        );
    }
    void DebugCellCounts()
    {
        uint[] cellCounts = new uint[CellRange];
        cellCountBuffer.GetData(cellCounts);

        for (int i = 0; i < CellRange; i++)
        {
            if (cellCounts[i] > 0)
            {
                Debug.Log($"Cell {i} : {cellCounts[i]} objects");
            }
        }        
    }

    void DebugVelocities(string label)
    {
        Vector3[] velocities = new Vector3[2];
        velocityBuffer.GetData(velocities);

        Debug.Log(
            $"{label} | " +
            $"A: {velocities[0]}, " +
            $"B: {velocities[1]}"
        );
    }
}
