using UnityEngine;

public class GPUCube_collision : MonoBehaviour
{

    public ComputeShader computeShader;
    public Material mat;
    public Mesh mesh;
    public uint count = 1000000;
    
    GraphicsBuffer commandBuffer;
    GraphicsBuffer.IndirectDrawIndexedArgs[] commandData;
    const int commandCount = 1;
    // Compute
    int mainKernel;
    GraphicsBuffer positionBuffer;
    GraphicsBuffer collisionBuffer;
    
    // Uniform-Grid
    int clearCellCountsKernel;    
    int clearCellWriteCountersKernel;    
    int updateCellCountsKernel;
    int updateObjectIDsKernel;
    int updateCellStartsKernel;
    int inclusiveToExclusiveKernel;
    

    GraphicsBuffer cellCountBuffer; 
    GraphicsBuffer objectIDBuffer;
    GraphicsBuffer cellStartBuffer;
    GraphicsBuffer cellWriteCounterBuffer;
    GraphicsBuffer prefixInputBuffer;
    GraphicsBuffer prefixOutputBuffer;
    // const int WorldMin = 25;
    const int GridSize = 50;
    const float CellSize = 4;
    const int CellRange = GridSize * GridSize * GridSize;


    void Start()
    {
        commandBuffer = new GraphicsBuffer(GraphicsBuffer.Target.IndirectArguments, commandCount, GraphicsBuffer.IndirectDrawIndexedArgs.size);
        commandData = new GraphicsBuffer.IndirectDrawIndexedArgs[commandCount];
        
        commandData[0].indexCountPerInstance = mesh.GetIndexCount(0);
        commandData[0].instanceCount = count;

        commandBuffer.SetData(commandData); 

        CreateBuffer();
        FindKernel();

        InitCubePositions();

        BindBuffer();
        BindUniformGridBuffer();
    }

    void CreateBuffer()
    {
        positionBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, (int)count, sizeof(float) * 3);
        collisionBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, (int)count, sizeof(uint));

        cellCountBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, CellRange, sizeof(uint));
        objectIDBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, (int)count, sizeof(uint));
        cellStartBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, CellRange, sizeof(uint));
        cellWriteCounterBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, CellRange, sizeof(uint));
        prefixInputBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, CellRange, sizeof(uint));
        prefixOutputBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, CellRange, sizeof(uint));
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
    }

    void BindBuffer()
    {
        computeShader.SetInt("_Count", (int)count);
        
        computeShader.SetInt("_GridSize", GridSize);
        computeShader.SetFloat("_CellSize", CellSize);

        computeShader.SetVector("_WorldMin", new Vector3(0,0,0));

        computeShader.SetBuffer(mainKernel, "_Positions", positionBuffer);
        computeShader.SetBuffer(mainKernel, "_Collisions", collisionBuffer);
        computeShader.SetBuffer(mainKernel, "_CellCounts", cellCountBuffer);
        computeShader.SetBuffer(mainKernel, "_CellStarts", cellStartBuffer);
        computeShader.SetBuffer(mainKernel, "_ObjectIDs", objectIDBuffer);

        mat.SetBuffer("_Positions", positionBuffer);
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
        
        // computeShader.SetBuffer(updateCellStartsKernel, "_CellCounts", cellCountBuffer);
        // computeShader.SetBuffer(updateCellStartsKernel, "_CellStarts", cellStartBuffer);
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
    }

    void Update()
    {
        computeShader.Dispatch(clearCellCountsKernel, Mathf.CeilToInt(CellRange / 64.0f), 1, 1);
        computeShader.Dispatch(clearCellWriteCountersKernel, Mathf.CeilToInt(CellRange / 64.0f), 1, 1);

        computeShader.Dispatch(updateCellCountsKernel, Mathf.CeilToInt(count / 64.0f), 1, 1);
        
        Hillis_Steele();

        computeShader.Dispatch(updateObjectIDsKernel, Mathf.CeilToInt(count / 64.0f), 1, 1);

        computeShader.Dispatch(mainKernel, Mathf.CeilToInt(count / 64.0f), 1, 1);

        RenderParams rp = new RenderParams(mat);
        rp.worldBounds = new Bounds(Vector3.zero, 10000*Vector3.one);        
        Graphics.RenderMeshIndirect(rp, mesh, commandBuffer, commandCount);

        // DebugCellCounts();
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

    void InitCubePositions()
    {
        Vector3[] positions = new Vector3[count];

        for (int i=0; i < count; i++)
        {
            int x = i % 100;
            int y = (i / 100) % 100;
            int z = i / 10000;

            positions[i] = new Vector3(x * 1, y * 1, z * 1);
        }

        positionBuffer.SetData(positions);
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
}
