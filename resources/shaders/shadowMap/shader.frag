#version 450

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragColor;
layout(location = 2) in vec3 fragNormal;
layout(location = 3) in vec2 fragTexCoord;
layout(location = 4) in vec4 fragPosFromLight;

layout(location = 0) out vec4 outColor;

layout(binding = 1) uniform sampler2D texSampler;
layout(binding = 2) uniform sampler2D shadowMap;



layout (binding  = 3) uniform UniformBufferObject {
	vec4 uCameraPos;
    vec4 uLightPos;
};

vec3 uLightColor = {1.0f, 1.0f, 1.0f};


// base light color
vec3 blinnPhong(float shadow)
{
	vec3 baseColor = texture(texSampler, fragTexCoord).rgb;

	vec3 norm = normalize(fragNormal);
	vec3 viewDir = normalize(uCameraPos.xyz - fragPos);
	vec3 lightDir = normalize(uLightPos.xyz - fragPos);

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
    vec3 viewDir = normalize(uCameraPos.xyz - fragPos);
    vec3 lightDir = normalize(uLightPos.xyz - fragPos);

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


// only shadowMap simple
float hardShadow(vec3 coord, float bias)
{
	float closestDepth = texture(shadowMap, coord.xy).r; 
    float currentDepth = coord.z;

    float shadow = currentDepth - bias > closestDepth  ? 1.0 : 0.0;
	if(coord.z > 1.0)
       shadow = 0.0;
	return shadow;
}

void main() {
	
	// to NDC
	vec3 shadowCoord = fragPosFromLight.xyz / fragPosFromLight.w;
	// shadowCoord = shadowCoord * 0.5 + 0.5;

	float shadow = hardShadow(shadowCoord, 0);

    outColor = vec4(blinnPhong(0), 1.0);
	// float closestDepth = texture(shadowMap, shadowCoord.xy).r;
    // outColor = vec4( closestDepth, closestDepth, closestDepth, 1.0);
}