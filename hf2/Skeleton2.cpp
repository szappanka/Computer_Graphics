//=============================================================================================
// Mintaprogram: Zold haromszog. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Szalka Panka
// Neptun : RITH1H
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL
const char* vertexSource = R"(
	#version 330
    precision highp float;

	uniform vec3 wLookAt, wRight, wUp;          // pos of eye

	layout(location = 0) in vec2 cCamWindowVertex;	// Attrib Array 0
	out vec3 p;

	void main() {
		gl_Position = vec4(cCamWindowVertex, 0, 1);
		p = wLookAt + wRight * cCamWindowVertex.x + wUp * cCamWindowVertex.y;
	}
)";
// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
    precision highp float;

	struct Material {
		vec3 ka, kd, ks;
		float  shininess;
		vec3 F0;
		int rough, reflective;
	};

	struct Light {
		vec3 direction;
		vec3 Le, La;
	};

	struct Hit {
		float t;
		vec3 position, normal;
		int mat;	// material index
	};

	struct Ray {
		vec3 start, dir;
	};

	const float epsilon = 0.0001f;

	uniform vec3 wEye; 
	uniform Light light;
	uniform Material materials[3];  // diffuse, specular, ambient ref
	uniform int planes[5*12];
	uniform vec3 v[20];

	in  vec3 p;					// point on camera window corresponding to the pixel
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation
	
	// a keplet: http://math.bme.hu/~wettl/okt/a1/2006/egyenessik.pdf
	float tavolsag(vec3 P1, vec3 P2, vec3 Q) {
	
		vec3 v = P1 - P2;  // iranyvektor
		vec3 P1_Q = Q - P1;  // PQ vektor
		float dist = length(cross(v, P1_Q))/length(v);

		return dist;
	}

	bool portal(vec3 point) {
		for (int i = 0; i < 12; i++) {
				for (int j = 0; j < 5; j++) {

					float dis = tavolsag(v[(planes[i * 5 + (j%5)] - 1)], v[(planes[i * 5 + ((j + 1)%5)] - 1)], point);
					if (dis < 0.1){
						return false;
					}
				}
			}
		return true;
	}

	// Csala Balint fele konzin hangzottal el az ehhez tartozo kepletek (t1 és t2 kiszamitasa a,b,c szamokbol
	Hit intersectHyperboloid(const Ray ray, Hit hit, int mat) {
		
		float a = 2.4;
		float b = 2.4;
		float c = 1.5;
		float A = a*ray.dir.x*ray.dir.x + b*ray.dir.y*ray.dir.y;
		float B = 2*a*ray.start.x*ray.dir.x + 2*b*ray.start.y*ray.dir.y - c*ray.dir.z;
		float C = a*ray.start.x*ray.start.x + b*ray.start.y*ray.start.y - c*ray.start.z;
		
		float discr = B * B - 4.0f * A * C;
		if (discr < 0) return hit;
		float sqrt_discr = sqrt(discr);
		float t1 = (-B + sqrt_discr) / 2.0f / A;	// t1 >= t2 for sure
		float t2 = (-B - sqrt_discr) / 2.0f / A;
		if (t1 <= 0) return hit;
		
		vec3 P1 = vec3(ray.start.x + ray.dir.x * t1, ray.start.y + ray.dir.y * t1, ray.start.z + ray.dir.z * t1);
		vec3 P2 = vec3(ray.start.x + ray.dir.x * t2, ray.start.y + ray.dir.y * t2, ray.start.z + ray.dir.z * t2);
		float dist_P1 = sqrt(P1.x*P1.x + P1.y*P1.y + P1.z*P1.z);
		float dist_P2 = sqrt(P2.x*P2.x + P2.y*P2.y + P2.z*P2.z);

		if(dist_P1 < 0.3f && dist_P2<0.3f){
			hit.t = (t2 > 0) ? t2 : t1;
		}
		else if(dist_P1 < 0.3f){
			hit.t = t1;
		}
		else if(dist_P2 < 0.3f){
			hit.t = t2;
		}
		else{
			return hit;
		}

		hit.position = ray.start + ray.dir * hit.t;
		vec3 der_x = vec3(1,0,(2*a*hit.position.x)/c);
		vec3 der_y = vec3(0,1,(2*b*hit.position.y)/c);
		hit.normal = normalize(cross(der_x, der_y));
		hit.mat = mat;
		return hit;
	}

	// 2020-as miasrascope-os hazi videojaban szerepelt
	void getPlane(int i, out vec3 p, out vec3 normal){

		vec3 p1 = v[planes[5*i]-1], p2 = v[planes[5*i+1]-1], p3 = v[planes[5*i+2]-1];
		normal = cross(p2-p1,p3-p1);
		if(dot(p1,normal)<0) normal = -normal;
		p = p1;
	}

	// 2020-as misrascope-os hazi videojaban szerepelt
	Hit intersectPolyhedron(Ray ray,Hit hit) {
		for(int i = 0; i < 12; i++){
			vec3 p1, normal;
			getPlane(i, p1, normal);
			float ti = abs(dot(normal, ray.dir)) > epsilon ? dot(p1-ray.start,normal)/dot(normal,ray.dir) : -1;
			if(ti<= epsilon || (ti > hit.t && hit.t>0)) continue;
			vec3 pintersect = ray.start + ray.dir * ti;
			bool outside = false;
			for(int j = 0; j < 12; j++){
				if(i==j) continue;
				vec3 p11, n;
				getPlane(j,p11,n);
				if(dot(n, pintersect - p11) > 0){
					outside = true;
					break;
				}
			}
			if(!outside){

				hit.t = ti;
				hit.position = pintersect;
				hit.normal = normalize(normal);

				if(portal(hit.position)){  
					hit.mat = 2;					
				}
				else{hit.mat = 0;}
			}
		}	
		return hit;
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		bestHit.t = -1;
		bestHit = intersectHyperboloid(ray, bestHit, 1);
		bestHit = intersectPolyhedron(ray, bestHit);
		
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights
		Hit hit;
		hit.t = -1;
		if (intersectHyperboloid(ray,hit, 1).t > 0) return true;
		return false;
	}

	vec3 Fresnel(vec3 F0, float cosTheta) { 
		return F0 + (vec3(1, 1, 1) - F0) * pow(cosTheta, 5);
	}

	vec4 qmul(vec4 q1, vec4 q2) {
		return vec4(q1.w * q2.xyz + q2.w * q1.xyz + cross(q1.xyz, q2.xyz), q1.w * q2.w - dot(q1.xyz, q2.xyz));
	} 

	vec4 quaternion(float ang, vec3 axis) {
		vec3 d = normalize(axis) * sin(ang / 2);
		return vec4(d.x, d.y, d.z, cos(ang / 2));
	}

	vec3 Rotate(vec3 u, vec4 q) {
		vec4 qinv = vec4(-q.x, -q.y, -q.z, q.w);
		vec4 qr = qmul(qmul(q, vec4(u.x, u.y, u.z, 0)), qinv);
		return qr.xyz;
	}
		
	const int maxdepth = 5;

	vec3 trace(Ray ray) {
		vec3 weight = vec3(1, 1, 1);
		vec3 outRadiance = vec3(0, 0, 0);
		for(int d = 0; d < maxdepth; d++) {
			Hit hit = firstIntersect(ray);

			if (hit.t < 0) return weight * light.La;
			
			if (materials[hit.mat].rough == 1) {
				outRadiance += weight * materials[hit.mat].ka * light.La;
				Ray shadowRay;
				shadowRay.start = hit.position + hit.normal * epsilon;
				shadowRay.dir = light.direction;
				float cosTheta = dot(hit.normal, light.direction);
				if (cosTheta > 0 && !shadowIntersect(shadowRay)) {
					outRadiance += weight * light.Le * materials[hit.mat].kd * cosTheta;
					vec3 halfway = normalize(-ray.dir + light.direction);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0) outRadiance += weight * light.Le * materials[hit.mat].ks * pow(cosDelta, materials[hit.mat].shininess);
				}
			}

			if (materials[hit.mat].reflective == 1) {
				weight *= Fresnel(materials[hit.mat].F0, dot(-ray.dir, hit.normal));
				ray.start = hit.position + hit.normal * epsilon;
				ray.dir = reflect(ray.dir, hit.normal);
				if(hit.mat==2){
			
					ray.start = Rotate(ray.start, quaternion(72.0*3.1415/180.0, hit.normal));
					ray.dir = Rotate(ray.dir, quaternion(72.0*3.1415/180.0, hit.normal));
				}
			} else return outRadiance;
		}
		return vec3(0.43,0.66,1);
	}

	void main() {
		Ray ray;
		ray.start = wEye; 
		ray.dir = normalize(p - wEye);
		fragmentColor = vec4(trace(ray), 1); 
	}
)";
// 2020-as misrascope-os hazi videojaban szerepelt
float F(float n, float k) { return((n - 1) * (n - 1) + k * k) / ((n + 1) * (n + 1) + k * k); }

//---------------------------
struct Material {
	//---------------------------
	vec3 ka, kd, ks;
	float  shininess;
	vec3 F0;
	int rough, reflective;
};

//---------------------------
struct RoughMaterial : Material {
	//---------------------------
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) {
		ka = _kd * M_PI;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
		rough = true;
		reflective = false;
	}
};

//---------------------------
struct SmoothMaterial : Material {
	//---------------------------
	SmoothMaterial(vec3 _F0) {
		F0 = _F0;
		rough = false;
		reflective = true;
	}
};

//---------------------------
struct Camera {
	//---------------------------
	vec3 eye, lookat, right, up;
	float fov;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
		eye = _eye;
		lookat = _lookat;
		fov = _fov;
		vec3 w = eye - lookat;
		float f = length(w);
		right = normalize(cross(vup, w)) * f * tanf(fov / 2);
		up = normalize(cross(w, right)) * f * tanf(fov / 2);
	}
	void Animate(float dt) {
		eye = vec3((eye.x - lookat.x) * cos(dt) + (eye.z - lookat.z) * sin(dt) + lookat.x,
			eye.y,
			-(eye.x - lookat.x) * sin(dt) + (eye.z - lookat.z) * cos(dt) + lookat.z);
		set(eye, lookat, up, fov);
	}
};

//---------------------------
struct Light {
	//---------------------------
	vec3 direction;
	vec3 Le, La;
	Light(vec3 _direction, vec3 _Le, vec3 _La) {
		direction = normalize(_direction);
		Le = _Le; La = _La;
	}
};

//---------------------------
class Shader : public GPUProgram {
	//---------------------------
public:
	void setUniformMaterials(const std::vector<Material*>& materials) {
		char name[256];
		for (unsigned int mat = 0; mat < materials.size(); mat++) {
			sprintf(name, "materials[%d].ka", mat); setUniform(materials[mat]->ka, name);
			sprintf(name, "materials[%d].kd", mat); setUniform(materials[mat]->kd, name);
			sprintf(name, "materials[%d].ks", mat); setUniform(materials[mat]->ks, name);
			sprintf(name, "materials[%d].shininess", mat); setUniform(materials[mat]->shininess, name);
			sprintf(name, "materials[%d].F0", mat); setUniform(materials[mat]->F0, name);
			sprintf(name, "materials[%d].rough", mat); setUniform(materials[mat]->rough, name);
			sprintf(name, "materials[%d].reflective", mat); setUniform(materials[mat]->reflective, name);
		}
	}

	void setUniformLight(Light* light) {
		setUniform(light->La, "light.La");
		setUniform(light->Le, "light.Le");
		setUniform(light->direction, "light.direction");
	}

	void setUniformCamera(const Camera& camera) {
		setUniform(camera.eye, "wEye");
		setUniform(camera.lookat, "wLookAt");
		setUniform(camera.right, "wRight");
		setUniform(camera.up, "wUp");
	}
};

//---------------------------
class Scene {
	//---------------------------
	std::vector<Light*> lights;
	Camera camera;
	std::vector<Material*> materials;
public:
	void build() {
		vec3 eye = vec3(0, 0, 1.3);
		vec3 vup = vec3(0, 1, 0);
		vec3 lookat = vec3(0, 0, 0);
		float fov = 48 * (float)M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		lights.push_back(new Light(vec3(1, 1, 1), vec3(3, 3, 3), vec3(0.4f, 0.3f, 0.3f)));

		vec3 kd(0.3f, 0.2f, 0.1f), ks(10, 10, 10);
		materials.push_back(new RoughMaterial(kd, ks, 100));
		materials.push_back(new SmoothMaterial(vec3(F(0.17f, 3.1f), F(0.35f, 2.7f), F(1.5f, 1.9f))));
		materials.push_back(new SmoothMaterial(vec3(1, 1, 1)));
	}

	void setUniform(Shader& shader) {
		shader.setUniformMaterials(materials);
		shader.setUniformLight(lights[0]);
		shader.setUniformCamera(camera);
	}

	void Animate(float dt) { camera.Animate(dt); }
};

Shader shader;
Scene scene;

//---------------------------
class FullScreenTexturedQuad {
	//---------------------------
	unsigned int vao = 0;
public:
	void create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);		

		unsigned int vbo;	
		glGenBuffers(1, &vbo);

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	}

	void Draw() {
		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	}
};




FullScreenTexturedQuad fullScreenTexturedQuad;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();
	fullScreenTexturedQuad.create();

	shader.create(vertexSource, fragmentSource, "fragmentColor");
	shader.Use();
	float x = 0.618, y = 1.618;
	// 2020-as misrascope-os hazi videojaban szerepelt
	std::vector<vec3> v = { vec3(0,x,y), vec3(0,-x,y), vec3(0,-x,-y), vec3(0,x,-y),
			vec3(y,0,x), vec3(-y,0,x), vec3(-y,0,-x), vec3(y,0,-x),
			vec3(x,y,0), vec3(-x,y,0), vec3(-x,-y,0), vec3(x,-y,0),
			vec3(1,1,1), vec3(-1,1,1), vec3(-1,-1,1), vec3(1,-1,1),
			vec3(1,-1,-1), vec3(1,1,-1), vec3(-1,1,-1), vec3(-1,-1,-1)
	};

	for (int i = 0; i < v.size(); i++) shader.setUniform(v[i], "v[" + std::to_string(i) + "]");

	std::vector<int> planes = {
		1, 2, 16, 5, 13 ,
		1, 13, 9, 10, 14,
		1, 14, 6, 15, 2,
		2, 15, 11, 12, 16,
		3, 4, 18, 8, 17,
		3, 17, 12, 11, 20,
		3, 20, 7, 19, 4,
		19, 10, 9, 18, 4,
		16, 12, 17, 8, 5,
		5, 8, 18, 9, 13,
		14, 10, 19, 7, 6,
		6, 7, 20, 11, 15
	};

	for (int i = 0; i < planes.size(); i++) shader.setUniform(planes[i], "planes[" + std::to_string(i) + "]");
}

void onDisplay() {

	glClearColor(1.0f, 0.5f, 0.8f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	scene.setUniform(shader);
	fullScreenTexturedQuad.Draw();

	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
}

void onKeyboardUp(unsigned char key, int pX, int pY) {

}

void onMouse(int button, int state, int pX, int pY) {
}

void onMouseMotion(int pX, int pY) {
}

void onIdle() {
	scene.Animate(0.01f);
	glutPostRedisplay();
}