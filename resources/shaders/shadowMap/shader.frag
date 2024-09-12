#version 450

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragColor;
layout(location = 2) in vec3 fragNormal;
layout(location = 3) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

layout(binding = 1) uniform sampler2D texSampler;


vec3 uLightColor = {1.0f, 1.0f, 1.0f};
vec3 uCameraPos = {2.0f, 2.0f, 2.0f};
vec3 uLightPos = {2.0f, 2.0f, 2.0f};

// base light color
vec3 blinnPhong(float shadow)
{
	vec3 baseColor = texture(texSampler, fragTexCoord).rgb;

	vec3 norm = normalize(fragNormal);
	vec3 viewDir = normalize(uCameraPos - fragPos);
	vec3 lightDir = normalize(uLightPos - fragPos);

	// ambient
	vec3 ambient = 0.3f * uLightColor;

	// diffuse
	float kd = max(dot(lightDir, norm), 0.0f);
	vec3 diffuse = kd * uLightColor;

	// specular
	vec3 halfv = normalize(lightDir + viewDir);
	float kp = pow( max( dot(halfv, norm), 0.0f), 64.0f);
	vec3 specular = kp * uLightColor;

	return (ambient + (1.0 - shadow)*(diffuse + specular)) * baseColor;
}

vec3 phong(float shadow)
{
	vec3 baseColor = texture(texSampler, fragTexCoord).rgb;
	
	vec3 norm = normalize(fragNormal);
    vec3 viewDir = normalize(uCameraPos - fragPos);
    vec3 lightDir = normalize(uLightPos - fragPos);

    // ambient
    vec3 ambient = 0.3f * uLightColor;

    // diffuse
    float kd  = max( dot(lightDir, norm), 0.0);
    vec3 diffuse = kd * uLightColor;

    // specular
    float spec = 0.05f;
    vec3 reflectDir = reflect(-lightDir, norm);
    float kp = pow( max( dot(viewDir, reflectDir), 0.0f), 32.0f );
    vec3 specular = spec * kp * uLightColor;
	
	return (ambient + (1.0 - shadow) * (diffuse + specular)) * baseColor;
}

void main() {
    outColor = vec4(phong(0.0f), 1.0);
}