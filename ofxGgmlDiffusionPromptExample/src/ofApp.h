#pragma once

#include "ofMain.h"
#include "ofxGgmlDiffusion.h"

class ofApp : public ofBaseApp {
public:
	void setup() override;
	void update() override;
	void keyPressed(int key) override;
	void draw() override;

private:
	void runGeneration();
	std::string findModelPath() const;
	std::string findPhotoMakerPath() const;
	bool loadPhotoMakerReferences();
	std::string getOutputPath() const;
	void applyResult(const ofxGgmlDiffusionResult& result);

	ofxGgmlDiffusionAsyncRunner runner;
	ofxGgmlDiffusionContextSettings settings;
	ofxGgmlDiffusionRequest request;
	ofTexture texture;
	std::string status;
	std::string detail;
};
