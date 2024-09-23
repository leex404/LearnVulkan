#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec2 inTexCoord;

layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 fragColor;
layout(location = 2) out vec3 fragNormal;
layout(location = 3) out vec2 fragTexCoord;
layout(location = 4) out vec4 fragPosFromLight;

layout (binding  = 0) uniform UniformBufferObject {
	mat4 view;
	mat4 proj;
	mat4 lightVP;
} ubo;

layout(push_constant) uniform PushConstants {
    mat4 model;
} pc;


const mat4 biasMat = mat4( 
	0.5, 0.0, 0.0, 0.0,
	0.0, 0.5, 0.0, 0.0,
	0.0, 0.0, 1.0, 0.0,
	0.5, 0.5, 0.0, 1.0 );

void main() {
    gl_Position = ubo.proj * ubo.view * pc.model * vec4(inPosition, 1.0);
	
	fragNormal = transpose(inverse(mat3(pc.model))) * normalize(inNormal);

	fragPos = vec3(pc.model * vec4(inPosition, 1.0));
    
	fragColor = inColor;
	fragTexCoord = inTexCoord;


	fragPosFromLight = (biasMat *  ubo.lightVP * pc.model ) * vec4(inPosition, 1.0);

}