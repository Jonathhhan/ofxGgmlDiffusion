#pragma once

#include "ofMain.h"
#include "ofxGgmlDiffusion.h"

#include <memory>
#include <string>

class ofApp : public ofBaseApp {
public:
	void setup() override;
	void keyPressed(int key) override;
	void draw() override;

private:
	void runGeneration();
	std::string findGeneratorPath() const;
	std::string getOutputPath() const;
	void rebuildRequest();

	ofxGgmlDiffusionContextSettings settings;
	ofxGgmlDiffusionRequest request;
	std::unique_ptr<ofxGgmlDiffusionImageGenerationBackend> backend;
	std::string prompt;
	std::string status;
	std::string detail;
};
