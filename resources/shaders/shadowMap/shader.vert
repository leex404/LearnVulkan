#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec2 inTexCoord;

layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 fragColor;
layout(location = 2) out vec3 fragNormal;
layout(location = 3) out vec2 fragTexCoord;

layout (binding  = 0) uniform UniformBufferObject {
	mat4 view;
	mat4 proj;
} ubo;

layout(push_constant) uniform PushConstants {
    mat4 model;
};

void main() {
    gl_Position = ubo.proj * ubo.view * model * vec4(inPosition, 1.0);
	
	fragNormal = transpose(inverse(mat3(model))) * inNormal;
	fragPos = vec3(model * vec4(inPosition, 1.0));
    fragColor = inColor;
	fragTexCoord = inTexCoord;
}