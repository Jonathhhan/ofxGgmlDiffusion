#pragma once

#include <string>
#include <vector>

struct ofxGgmlDiffusionTinyGanTrainingSettings {
	std::string datasetPath;
	std::string outputPresetPath;
	int imageWidth = 64;
	int imageHeight = 64;
	int latentSize = 512;
	int hiddenSize = 96;
	int epochs = 1;
	int batchSize = 4;
	float learningRate = 0.001f;
	bool dryRun = true;
};

struct ofxGgmlDiffusionTinyGanTrainingResult {
	bool success = false;
	std::string text;
	std::string error;
	std::string outputPresetPath;
	int epochsPlanned = 0;
	int batchSize = 0;
	float learningRate = 0.0f;
	std::vector<std::string> warnings;

	bool isOk() const {
		return success;
	}

	bool isError() const {
		return !success;
	}

	explicit operator bool() const {
		return success;
	}
};

ofxGgmlDiffusionTinyGanTrainingResult ofxGgmlDiffusionValidateTinyGanTraining(
	const ofxGgmlDiffusionTinyGanTrainingSettings& settings);
ofxGgmlDiffusionTinyGanTrainingResult ofxGgmlDiffusionPlanTinyGanTraining(
	const ofxGgmlDiffusionTinyGanTrainingSettings& settings);
ofxGgmlDiffusionTinyGanTrainingResult ofxGgmlDiffusionRunTinyGanTraining(
	const ofxGgmlDiffusionTinyGanTrainingSettings& settings);
