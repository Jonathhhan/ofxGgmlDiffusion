#pragma once

#include "ofMain.h"
#include "ofxGgmlDiffusion.h"

class ofApp : public ofBaseApp {
public:
	void setup() override;
	void keyPressed(int key) override;
	void draw() override;

private:
	void runGeneration();
	std::string findModelPath() const;
	std::string getOutputPath() const;

	ofxGgmlDiffusionNativeBackend backend;
	ofxGgmlDiffusionContextSettings settings;
	ofxGgmlDiffusionRequest request;
	ofTexture texture;
	std::string status;
	std::string detail;
};
