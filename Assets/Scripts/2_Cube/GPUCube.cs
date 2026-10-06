using UnityEngine;

public class GPUCube : MonoBehaviour
{
    public Material mat;
    public Mesh mesh;

    GraphicsBuffer commandBuffer;
    GraphicsBuffer.IndirectDrawIndexedArgs[] commandData;
    const int commandCount = 1;

    void Start()
    {
        commandBuffer = new GraphicsBuffer(GraphicsBuffer.Target.IndirectArguments, commandCount, GraphicsBuffer.IndirectDrawIndexedArgs.size);
        commandData = new GraphicsBuffer.IndirectDrawIndexedArgs[commandCount];
        
        commandData[0].indexCountPerInstance = mesh.GetIndexCount(0);
        commandData[0].instanceCount = 1000000;

        commandBuffer.SetData(commandData); 
    }

    void OnDestroy()
    {
        commandBuffer?.Release();
        commandBuffer = null;
    }

    void Update()
    {
        RenderParams rp = new RenderParams(mat);
        rp.worldBounds = new Bounds(Vector3.zero, 10000*Vector3.one);
        
        Graphics.RenderMeshIndirect(rp, mesh, commandBuffer, commandCount);
    }
}
