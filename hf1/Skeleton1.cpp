//=============================================================================================
// Mintaprogram: Z?ld h?romsz?g. Ervenyes 2019. osztol.
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

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char* const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers
 
	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0
	layout(location = 1) in vec2 vtxUV;
 
	out vec2 texcoord;
 
	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
		texcoord = vtxUV;
	}
)";

// fragment shader in GLSL
const char* const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	
	uniform sampler2D samplerUnit;
	in vec2 texcoord;
	out vec4 fragmentColor;		// computed color of the current pixel
 
	void main() {
		fragmentColor = texture(samplerUnit, texcoord);	// computed color is the color of the primitive
	}
)";


struct Node {

	vec3 node;
	vec3 vonz;

public:
	Node() {};
	Node(float xx, float yy) { node.x = xx; node.y = yy; node.z = sqrt(1 + pow(xx, 2) + pow(yy, 2)); vonz.x = xx; vonz.y = yy; vonz.z = sqrt(1 + xx * xx + yy * yy); };
	Node(float xx, float yy, float zz) { node.x = xx; node.y = yy; node.z = zz; vonz.x = xx; vonz.y = yy; vonz.z = zz; };

	bool operator==(Node n) const {
		if (n.node.x == this->node.x && n.node.y == this->node.y) { return true; }
		return false;
	}

	float distance_eu(Node n) {
		return sqrt(pow(n.node.x - node.x, 2) + pow(n.node.y - node.y, 2));
	}

	float distance_hip(Node n) {

		float dist = acosh(-((node.x * n.node.x) + (node.y * n.node.y) - (node.z * n.node.z)));
		return dist;
	}

	vec3 vector(Node n) {

		vec3 v = (n.node - node * cosh(distance_hip(n))) / sinh(distance_hip(n));
		return normalize(v);
	}

	void change(vec3 v, float distance) {

		vonz.x = (vonz.x * cosh(distance) + v.x * sinh(distance));
		vonz.y = (vonz.y * cosh(distance) + v.y * sinh(distance));
		vonz.z = sqrt(1 + pow(vonz.x, 2) + pow(vonz.y, 2));
	}

	float difference() {

		return acosh(-((node.x * vonz.x) + (node.y * vonz.y) - (node.z * vonz.z)));
	}

	void mirror(Node n1, Node n2) {

		node.x = node.x + (2 * (n1.node.x - node.x));
		node.y = node.y + (2 * (n1.node.y - node.y));

		node.x = node.x + (2 * (n2.node.x - node.x));
		node.y = node.y + (2 * (n2.node.y - node.y));
		node.z = sqrt(1 + pow(node.x, 2) + pow(node.y, 2));

		vonz.x = node.x;
		vonz.y = node.y;
		vonz.z = node.z;
	}
};

struct Link {

	Node node1;
	Node node2;

public:
	Link() {};
	Link(Node n1, Node n2) { node1.node.x = n1.node.x; node1.node.y = n1.node.y; node2.node.x = n2.node.x; node2.node.y = n2.node.y; };

	bool operator==(Link l) const {

		if (l.node1 == this->node1 && l.node2 == this->node2) { return true; }
		else if (l.node2 == this->node1 && l.node1 == this->node2) { return true; }

		return false;
	}
};

Node nodes[50];
Link links[61];

GPUProgram gpuProgram;
unsigned int vao;
float oldMouse_x, oldMouse_y;

float links_draw[4 * 61] = {};
vec2 circle[100];
float d = 0.4;
bool simulation = false;

unsigned int texture;
vec4 szin_init[50];

void uploadTexture(int width, int height, vec4 im) {

	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA,
		GL_FLOAT, &im);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

}


bool intersection(Link l1, Link l2) {

	float A1 = l1.node2.node.y - l1.node1.node.y;
	float B1 = l1.node1.node.x - l1.node2.node.x;
	float C1 = (A1 * l1.node1.node.x) + (B1 * l1.node1.node.y);

	float A2 = l2.node2.node.y - l2.node1.node.y;
	float B2 = l2.node1.node.x - l2.node2.node.x;
	float C2 = (A2 * l2.node1.node.x) + (B2 * l2.node1.node.y);

	float x, y;

	y = ((C2 * A1) - (C1 * A2)) / ((A1 * B2) - (A2 * B1));
	x = (C1 - (B1 * y)) / A1;

	float minx1 = (l1.node1.node.x < l1.node2.node.x) ? l1.node1.node.x : l1.node2.node.x;
	float maxx1 = (l1.node1.node.x > l1.node2.node.x) ? l1.node1.node.x : l1.node2.node.x;
	float minx2 = (l2.node1.node.x < l2.node2.node.x) ? l2.node1.node.x : l2.node2.node.x;
	float maxx2 = (l2.node1.node.x > l2.node2.node.x) ? l2.node1.node.x : l2.node2.node.x;

	float miny1 = (l1.node1.node.y < l1.node2.node.y) ? l1.node1.node.y : l1.node2.node.y;
	float maxy1 = (l1.node1.node.y > l1.node2.node.y) ? l1.node1.node.y : l1.node2.node.y;
	float miny2 = (l2.node1.node.y < l2.node2.node.y) ? l2.node1.node.y : l2.node2.node.y;
	float maxy2 = (l2.node1.node.y > l2.node2.node.y) ? l2.node1.node.y : l2.node2.node.y;

	if (x > minx1 && x < maxx1
		&& x > minx2 && x < maxx2
		&& y > miny1 && y < maxy1
		&& y > miny2 && y < maxy2) {
		return true;
	}

	if (l1 == l2) { return true; }

	return false;
}

void betolt() {

	for (int i = 0; i < 61; i++) {

		float w1 = links[i].node1.node.z;
		float w2 = links[i].node2.node.z;

		links_draw[4 * i] = links[i].node1.node.x / links[i].node1.node.z;
		links_draw[4 * i + 1] = links[i].node1.node.y / links[i].node1.node.z;
		links_draw[4 * i + 2] = links[i].node2.node.x / links[i].node2.node.z;
		links_draw[4 * i + 3] = links[i].node2.node.y / links[i].node2.node.z;
	}
}


void szin_initialize() {

	for (int i = 0; i < 50; i++) {

		float x = ((float)rand()) / (float)RAND_MAX;
		float y = ((float)rand()) / (float)RAND_MAX;
		float z = ((float)rand()) / (float)RAND_MAX;
		float q = ((float)rand()) / (float)RAND_MAX;

		szin_init[i] = vec4(x, y, z, q);
	}
}

//unsigned int vbo[2];
// Initialization, create an OpenGL context
void onInitialization() {

	glViewport(0, 0, windowWidth, windowHeight);

	glGenVertexArrays(1, &vao);	// get 1 vao id
	glBindVertexArray(vao);		// make it active

	unsigned int vbo;		// vertex buffer object
	glGenBuffers(1, &vbo);

	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	// Geometry with 24 bytes (6 floats or 3 x 2 coordinates)

	for (int i = 0; i < 50; i++)
	{
		float random = ((float)rand()) / (float)RAND_MAX;
		float r = random * (1.0f - (-1.0));
		float done_x = (-1.0f) + r;

		float random2 = ((float)rand()) / (float)RAND_MAX;
		float r2 = random2 * (1.0f - (-1.0));
		float done_y = (-1.0f) + r2;

		nodes[i] = Node(done_x, done_y);

		for (int j = 0; j < i; j++)
		{
			if (nodes[i] == nodes[j] || nodes[i].distance_eu(nodes[j]) < 0.2) {
				i--;
				break;
			}
		}
	}

	for (int i = 0; i < 61; i++) {

		int t = i;
		int random = ((int)rand() % 50);
		int random2 = ((int)rand() % 50);

		if (random == random2) {
			t--;
		}

		Link temp = Link(nodes[random], nodes[random2]);

		if (nodes[random].distance_eu(nodes[random2]) > 0.8) {
			t--;
		}

		for (int k = 0; k < i; k++) {

			if (temp == links[k]) {
				t--;
				break;
			}

			if (intersection(temp, links[k])) {
				t--;
				break;
			}
		}

		if (t == i) {

			links[i] = Link(nodes[random], nodes[random2]);
			links[i].node1.node.z = sqrt(1 + links[i].node1.node.x * links[i].node1.node.x + links[i].node1.node.y * links[i].node1.node.y);
			links[i].node2.node.z = sqrt(1 + links[i].node2.node.x * links[i].node2.node.x + links[i].node2.node.y * links[i].node2.node.y);
		}
		else {
			i--;
		}
	}

	betolt();


	szin_initialize();

	glEnableVertexAttribArray(0);  // AttribArray 0
	glVertexAttribPointer(0,       // vbo -> AttribArray 0
		2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
		0, NULL); 		     // stride, offset: tightly packed

		// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0.25, 0.25, 0.25, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

	// Set color to (0, 1, 0) = green
	int location = glGetUniformLocation(gpuProgram.getId(), "color");
	glUniform3f(location, 0.0f, 0.0f, 0.0f); // 3 floats


	float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
							  0, 1, 0, 0,    // row-major!
							  0, 0, 1, 0,
							  0, 0, 0, 1 };

	location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
	glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location

	glBindVertexArray(vao);  // Draw call

	uploadTexture(1, 1, vec4(0.0, 0.0, 0.0, 0.0));

	glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
		sizeof(links_draw),  // # bytes
		links_draw,	      	// address
		GL_STATIC_DRAW);	// we do not change later

	glDrawArrays(GL_LINES, 0 /*startIdx*/, 2 * 61 /*# Elements*/);

	location = glGetUniformLocation(gpuProgram.getId(), "color");


	glUniform3f(location, 1.0f, 0.8f, 0.64f); // 3 floats



	for (int i = 0; i < 50; i++) {

		for (int j = 0; j < 100; j++) {

			float r = 0.03;
			float fi = j * 2 / M_PI;

			Node temp = Node(cosf(fi) * r + nodes[i].node.x, sinf(fi) * r + nodes[i].node.y);

			Node newNode = Node(nodes[i].node.x, nodes[i].node.y);

			newNode.change(newNode.vector(temp), r);

			newNode = Node(newNode.vonz.x, newNode.vonz.y);

			circle[j] = vec2(newNode.node.x / newNode.node.z, newNode.node.y / newNode.node.z);
		}




		uploadTexture(1, 1, szin_init[i]);

		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vec2) * 100,  // # bytes
			circle,	      	// address
			GL_STATIC_DRAW);

		int sampler = 0;

		int location = glGetUniformLocation(gpuProgram.getId(), "samplerUnit");
		glUniform1i(location, sampler);

		glActiveTexture(GL_TEXTURE0 + sampler);
		glBindTexture(GL_TEXTURE_2D, texture);

		glDrawArrays(GL_TRIANGLE_FAN, 0 /*startIdx*/, 100 /*# Elements*/);

	}

	glutSwapBuffers(); // exchange buffers for double buffering
}

bool linkInArray(Link l) {

	for (int i = 0; i < 61; i++) {

		if (l == links[i]) {
			return true;
		}
	}
	return false;
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
	else if (key == ' ') simulation = true;
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space

	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	Node temp = Node(cX, cY);

	Node p1 = Node(oldMouse_x, oldMouse_y);
	float dis = p1.distance_hip(temp) / 2;
	vec3 v = p1.vector(temp);
	Node p2 = Node(p1.node.x * cosh(dis) + v.x * sinh(dis), p1.node.y * cosh(dis) + v.y * sinh(dis));

	if (!isnan(p2.node.x) && !isnan(p2.node.y) && !isnan(p2.node.z)) {
		for (int i = 0; i < 50; i++)
		{
			nodes[i].mirror(p1, p2);
		}
		for (int i = 0; i < 61; i++)
		{
			links[i].node1.mirror(p1, p2);
			links[i].node2.mirror(p1, p2);
		}
	}

	oldMouse_x = cX;
	oldMouse_y = cY;

	betolt();
	onDisplay();
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	oldMouse_x = cX;
	oldMouse_y = cY;
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {

	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program

	if (simulation) {

		for (int i = 0; i < 50; i++)
		{
			Node origo = Node(0.0, 0.0);
			float dist = nodes[i].distance_hip(origo);
			float scale = dist;

			nodes[i].change(nodes[i].vector(origo), dist * scale / 10);

			for (int j = 0; j < 50; j++)
			{
				if (!(nodes[i] == nodes[j]) && nodes[i].distance_hip(nodes[j]) > 0.0) {

					if (linkInArray(Link(nodes[i], nodes[j]))) {

						float dist = nodes[i].distance_hip(nodes[j]);
						float scale = tan(dist - d);

						nodes[i].change(nodes[i].vector(nodes[j]), dist * scale / 2);
					}

					else {
						if (nodes[i].distance_hip(nodes[j]) < 3.5) {
							float dist = nodes[i].distance_hip(nodes[j]);
							float scale = (-1 / (40 * (dist))) + 0.02;

							nodes[i].change(nodes[i].vector(nodes[j]), dist * scale);

						}
					}
				}
			}
		}

		for (int i = 0; i < 50; i++) {

			for (int j = 0; j < 61; j++)
			{
				if (nodes[i] == links[j].node1) {

					links[j].node1 = Node(nodes[i].vonz.x, nodes[i].vonz.y);
				}

				else if (nodes[i] == links[j].node2) {

					links[j].node2 = Node(nodes[i].vonz.x, nodes[i].vonz.y);
				}
			}

			nodes[i] = Node(nodes[i].vonz.x, nodes[i].vonz.y);

			int temp = 0;

			for (int k = 0; k < 50; k++) {
				if (nodes[k].difference() < 0.0005)
				{
					temp++;
				}
			}

			if (temp > 40) { simulation = false; }
		}

		betolt();
		onDisplay();
	}
}