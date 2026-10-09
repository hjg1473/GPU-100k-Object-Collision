Shader "Custom/7_Cube"
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
            StructuredBuffer<float4> _Rotations;

            StructuredBuffer<uint> _Collisions;

            float Random(uint seed)
            {
                return frac(sin(seed) * 12345.6789);
            }

            float3 RotateByQuaternion(float3 v, float4 q)
            {
                float3 t = 2.0f * cross(q.xyz, v);
                return v + q.w * t + cross(q.xyz, t);
            }

            Varyings vert(Attributes IN, uint svInstanceID : SV_InstanceID)
            {
                Varyings OUT;

                InitIndirectDrawArgs(0);

                uint instanceID = GetIndirectInstanceID(svInstanceID);
                
                float3 pos = IN.positionOS.xyz;

                float4 rot = _Rotations[instanceID];
                pos = RotateByQuaternion(pos, rot);
                
                pos += _Positions[instanceID];

                OUT.positionHCS = TransformObjectToHClip(pos);
                OUT.uv = TRANSFORM_TEX(IN.uv, _BaseMap);
                OUT.color = float3(Random(instanceID + 0), Random(instanceID + 1), Random(instanceID + 2));

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
