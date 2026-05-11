#include "ofxGgmlDiffusionTinyGanTraining.h"

#include <cmath>
#include <sstream>

namespace {
	ofxGgmlDiffusionTinyGanTrainingResult makeTrainingError(const std::string& message) {
		ofxGgmlDiffusionTinyGanTrainingResult result;
		result.success = false;
		result.error = message;
		return result;
	}

	void appendError(std::string& errors, const std::string& message) {
		if (!errors.empty()) {
			errors += "; ";
		}
		errors += message;
	}

	bool isFinitePositive(float value) {
		return std::isfinite(value) && value > 0.0f;
	}
}

ofxGgmlDiffusionTinyGanTrainingResult ofxGgmlDiffusionValidateTinyGanTraining(
	const ofxGgmlDiffusionTinyGanTrainingSettings& settings) {
	std::string errors;
	if (settings.imageWidth != 64 || settings.imageHeight != 64) {
		appendError(errors, "tiny GAN training currently supports 64x64 images only");
	}
	if (settings.latentSize < 8 || settings.latentSize > 1024) {
		appendError(errors, "latentSize must be between 8 and 1024");
	}
	if (settings.hiddenSize < 8 || settings.hiddenSize > 512) {
		appendError(errors, "hiddenSize must be between 8 and 512");
	}
	if (settings.discriminatorHiddenSize < 8 || settings.discriminatorHiddenSize > 512) {
		appendError(errors, "discriminatorHiddenSize must be between 8 and 512");
	}
	if (settings.epochs < 1) {
		appendError(errors, "epochs must be at least 1");
	}
	if (settings.batchSize < 1) {
		appendError(errors, "batchSize must be at least 1");
	}
	if (settings.dryRunBatchesPerEpoch < 1 || settings.dryRunBatchesPerEpoch > 128) {
		appendError(errors, "dryRunBatchesPerEpoch must be between 1 and 128");
	}
	if (!isFinitePositive(settings.learningRate)) {
		appendError(errors, "learningRate must be finite and greater than zero");
	}
	if (settings.outputPresetPath.empty()) {
		appendError(errors, "outputPresetPath is required");
	}
	if (!settings.dryRun) {
		appendError(errors, "tiny GAN training currently supports dry-run planning only");
	}

	if (!errors.empty()) {
		return makeTrainingError(errors);
	}

	ofxGgmlDiffusionTinyGanTrainingResult result;
	result.success = true;
	result.text = "tiny GAN training settings are valid";
	result.outputPresetPath = settings.outputPresetPath;
	result.epochsPlanned = settings.epochs;
	result.batchSize = settings.batchSize;
	result.batchesPerEpoch = settings.dryRunBatchesPerEpoch;
	result.plannedDiscriminatorUpdates = settings.epochs * settings.dryRunBatchesPerEpoch;
	result.plannedGeneratorUpdates = settings.epochs * settings.dryRunBatchesPerEpoch;
	result.learningRate = settings.learningRate;
	if (settings.datasetPath.empty()) {
		result.warnings.push_back("no dataset path provided; planning uses a synthetic dataset placeholder");
	}
	return result;
}

std::vector<ofxGgmlDiffusionTinyGanTrainingStep> ofxGgmlDiffusionBuildTinyGanTrainingDryRunSteps(
	const ofxGgmlDiffusionTinyGanTrainingSettings& settings) {
	std::vector<ofxGgmlDiffusionTinyGanTrainingStep> steps;
	const int totalSteps = settings.epochs * settings.dryRunBatchesPerEpoch;
	if (totalSteps <= 0) {
		return steps;
	}

	steps.reserve(static_cast<std::size_t>(totalSteps));
	for (int epoch = 0; epoch < settings.epochs; ++epoch) {
		for (int batch = 0; batch < settings.dryRunBatchesPerEpoch; ++batch) {
			const int index = epoch * settings.dryRunBatchesPerEpoch + batch;
			const float progress = static_cast<float>(index + 1) / static_cast<float>(totalSteps);
			ofxGgmlDiffusionTinyGanTrainingStep step;
			step.epoch = epoch + 1;
			step.batch = batch + 1;
			step.discriminatorLoss = 1.3862944f - (0.18f * progress);
			step.generatorLoss = 0.6931472f - (0.07f * progress);
			steps.push_back(step);
		}
	}
	return steps;
}

ofxGgmlDiffusionTinyGanTrainingResult ofxGgmlDiffusionPlanTinyGanTraining(
	const ofxGgmlDiffusionTinyGanTrainingSettings& settings) {
	auto result = ofxGgmlDiffusionValidateTinyGanTraining(settings);
	if (!result) {
		return result;
	}
	result.steps = ofxGgmlDiffusionBuildTinyGanTrainingDryRunSteps(settings);
	if (!result.steps.empty()) {
		result.finalDiscriminatorLoss = result.steps.back().discriminatorLoss;
		result.finalGeneratorLoss = result.steps.back().generatorLoss;
	}

	std::ostringstream text;
	text << "tiny GAN training dry-run\n";
	text << "dataset: " << (settings.datasetPath.empty() ? "(synthetic placeholder)" : settings.datasetPath) << "\n";
	text << "output preset: " << settings.outputPresetPath << "\n";
	text << "generator architecture: tiny-mlp\n";
	text << "discriminator architecture: tiny-mlp-binary-classifier\n";
	text << "image size: " << settings.imageWidth << "x" << settings.imageHeight << "\n";
	text << "latent size: " << settings.latentSize << "\n";
	text << "generator hidden size: " << settings.hiddenSize << "\n";
	text << "discriminator hidden size: " << settings.discriminatorHiddenSize << "\n";
	text << "epochs: " << settings.epochs << "\n";
	text << "batch size: " << settings.batchSize << "\n";
	text << "dry-run batches per epoch: " << settings.dryRunBatchesPerEpoch << "\n";
	text << "planned discriminator updates: " << result.plannedDiscriminatorUpdates << "\n";
	text << "planned generator updates: " << result.plannedGeneratorUpdates << "\n";
	text << "learning rate: " << settings.learningRate << "\n";
	text << "final dry-run discriminator loss: " << result.finalDiscriminatorLoss << "\n";
	text << "final dry-run generator loss: " << result.finalGeneratorLoss << "\n";
	text << "status: validated planning only; no weights were trained";
	result.text = text.str();
	return result;
}

ofxGgmlDiffusionTinyGanTrainingResult ofxGgmlDiffusionRunTinyGanTraining(
	const ofxGgmlDiffusionTinyGanTrainingSettings& settings) {
	if (!settings.dryRun) {
		return makeTrainingError(
			"tiny GAN training is experimental and currently supports dry-run planning only");
	}
	return ofxGgmlDiffusionPlanTinyGanTraining(settings);
}
