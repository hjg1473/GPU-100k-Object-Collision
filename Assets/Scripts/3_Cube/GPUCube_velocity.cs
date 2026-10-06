using UnityEngine;

public class GPUCube_velocity : MonoBehaviour
{

    public ComputeShader computeShader;
    public Material mat;
    public Mesh mesh;
    public uint count = 1000000;
    
    
    GraphicsBuffer commandBuffer;
    GraphicsBuffer.IndirectDrawIndexedArgs[] commandData;
    const int commandCount = 1;


    GraphicsBuffer positionBuffer;
    int calculateVelocityKernel;

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
    }

    void CreateBuffer()
    {
        positionBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, (int)count, sizeof(float) * 3);
    }

    void FindKernel()
    {
        calculateVelocityKernel = computeShader.FindKernel("calculateVelocity");
    }

    void BindBuffer()
    {
        computeShader.SetFloat("_Count", count);
        computeShader.SetBuffer(calculateVelocityKernel, "_Positions", positionBuffer);

        mat.SetBuffer("_Positions", positionBuffer);
    }

    void OnDestroy()
    {
        commandBuffer?.Release();
        commandBuffer = null;

        positionBuffer?.Release();
        positionBuffer = null;
    }

    void Update()
    {
        computeShader.Dispatch(calculateVelocityKernel, Mathf.CeilToInt(count / 64.0f), 1, 1);

        RenderParams rp = new RenderParams(mat);
        rp.worldBounds = new Bounds(Vector3.zero, 10000*Vector3.one);
        
        Graphics.RenderMeshIndirect(rp, mesh, commandBuffer, commandCount);
    }

    void InitCubePositions()
    {
        Vector3[] positions = new Vector3[count];

        for (int i=0; i < count; i++)
        {
            int x = i % 100;
            int y = (i / 100) % 100;
            int z = i / 10000;

            positions[i] = new Vector3(x * 2, y * 2, z * 2);
        }

        positionBuffer.SetData(positions);
    }

}
