Shader "Custom/4_Cube"
{
    Properties
    {
        [MainColor] _BaseColor("Base Color", Color) = (1, 1, 1, 1)
        [MainTexture] _BaseMap("Base Map", 2D) = "white" {}
    }

    SubShader
    {
        Tags { "RenderType" = "Opaque" "RenderPipeline" = "UniversalPipeline" }

        Pass
        {
            HLSLPROGRAM

            #pragma vertex vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            #define UNITY_INDIRECT_DRAW_ARGS IndirectDrawIndexedArgs
            #include "UnityIndirect.cginc"

            struct Attributes
            {
                float4 positionOS : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float2 uv : TEXCOORD0;
                float3 color : TEXCOORD1;
            };

            TEXTURE2D(_BaseMap);
            SAMPLER(sampler_BaseMap);

            CBUFFER_START(UnityPerMaterial)
                half4 _BaseColor;
                float4 _BaseMap_ST;
            CBUFFER_END

            StructuredBuffer<float3> _Positions;
            StructuredBuffer<uint> _Collisions;

            float Random(uint seed)
            {
                return frac(sin(seed) * 12345.6789);
            }

            Varyings vert(Attributes IN, uint svInstanceID : SV_InstanceID)
            {
                Varyings OUT;

                InitIndirectDrawArgs(0);

                uint instanceID = GetIndirectInstanceID(svInstanceID);
                
                float3 pos = IN.positionOS.xyz;
                
                pos += _Positions[instanceID];

                OUT.positionHCS = TransformObjectToHClip(pos);
                OUT.uv = TRANSFORM_TEX(IN.uv, _BaseMap);
                OUT.color = (_Collisions[instanceID] == 1) ? float3(1,0,0) : float3(0.2,0.2,0.2); 

                return OUT;
            }

            half4 frag(Varyings IN) : SV_Target
            {
                half4 color = SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, IN.uv) * _BaseColor;

                color.rgb *= IN.color;

                return color;
            }
            ENDHLSL
        }
    }
}
