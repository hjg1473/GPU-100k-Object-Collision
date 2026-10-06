using UnityEngine;

public class First_Cube : MonoBehaviour
{
    public Material mat;
    public Mesh mesh;
    public int count = 100;

    struct CubeData
    {
        public Matrix4x4 objectToWorld;
        public uint renderingLayerMask;
    }

    private CubeData[] instData;

    void Start()
    {        
        instData = new CubeData[count * count * count];
        
        int index = 0;

        for (int i=0; i<count; i++)
            for (int j=0; j<count; j++)
                for (int k=0; k<count; k++)
                {
                    instData[index].objectToWorld = Matrix4x4.Translate(new Vector3(i * 2f, j * 2f, k * 2f));
                    instData[index].renderingLayerMask = 0;

                    index++;
                }
    }
    
    void Update()
    {
        RenderParams rp = new RenderParams(mat);

        Graphics.RenderMeshInstanced(rp, mesh, 0, instData);
            
    }
}
