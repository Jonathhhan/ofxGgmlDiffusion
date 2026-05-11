#include "ofxGgmlDiffusionTinyGanTraining.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>

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

	std::string toLower(std::string value) {
		std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
			return static_cast<char>(std::tolower(c));
		});
		return value;
	}

	bool isSupportedImageExtension(const std::filesystem::path& path) {
		const auto extension = toLower(path.extension().string());
		return extension == ".png" ||
			extension == ".jpg" ||
			extension == ".jpeg" ||
			extension == ".bmp" ||
			extension == ".tga" ||
			extension == ".ppm";
	}

	int fixtureChannelValue(int x, int y, int index, int channel) {
		const int stripe = ((x / 8) + index + channel) % 2;
		const int checker = ((x / 8) + (y / 8) + index) % 2;
		const int gradient = (x * 3 + y * 5 + index * 29 + channel * 47) % 256;
		if (index % 3 == 0) {
			return stripe ? 224 : 32;
		}
		if (index % 3 == 1) {
			return checker ? 196 : 56;
		}
		return gradient;
	}

	bool readPpmToken(std::istream& input, std::string& token) {
		while (input >> token) {
			if (!token.empty() && token[0] == '#') {
				input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
				continue;
			}
			return true;
		}
		return false;
	}

	bool parsePpmInt(std::istream& input, int& value) {
		std::string token;
		if (!readPpmToken(input, token)) {
			return false;
		}
		try {
			std::size_t used = 0;
			const int parsed = std::stoi(token, &used);
			if (used != token.size()) {
				return false;
			}
			value = parsed;
			return true;
		} catch (...) {
			return false;
		}
	}

	float sigmoid(float value) {
		if (value >= 0.0f) {
			const float z = std::exp(-value);
			return 1.0f / (1.0f + z);
		}
		const float z = std::exp(value);
		return z / (1.0f + z);
	}

	float fixedTinyDiscriminatorWeight(std::size_t inputIndex, int hiddenIndex) {
		const int bucket = static_cast<int>(((inputIndex + 1) * static_cast<std::size_t>(hiddenIndex + 3)) % 23);
		return static_cast<float>(bucket - 11) / 96.0f;
	}

	float clampProbability(float probability) {
		if (!std::isfinite(probability)) {
			return 0.5f;
		}
		return std::clamp(probability, 0.000001f, 0.999999f);
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

ofxGgmlDiffusionTinyGanDatasetScan ofxGgmlDiffusionScanTinyGanDataset(
	const std::string& datasetPath) {
	ofxGgmlDiffusionTinyGanDatasetScan scan;
	scan.path = datasetPath;
	if (datasetPath.empty()) {
		scan.warnings.push_back("no dataset path provided; planning uses a synthetic dataset placeholder");
		return scan;
	}

	const std::filesystem::path root(datasetPath);
	std::error_code error;
	scan.exists = std::filesystem::exists(root, error);
	if (error || !scan.exists) {
		scan.warnings.push_back("dataset path does not exist");
		return scan;
	}

	scan.isDirectory = std::filesystem::is_directory(root, error);
	if (error || !scan.isDirectory) {
		scan.warnings.push_back("dataset path is not a directory");
		return scan;
	}

	for (const auto& entry : std::filesystem::recursive_directory_iterator(root, error)) {
		if (error) {
			scan.warnings.push_back("dataset scan stopped early because a directory could not be read");
			break;
		}
		if (!entry.is_regular_file(error) || error) {
			error.clear();
			continue;
		}
		if (isSupportedImageExtension(entry.path())) {
			++scan.imageCount;
			scan.imagePaths.push_back(entry.path().string());
		} else {
			++scan.unsupportedFileCount;
		}
	}

	std::sort(scan.imagePaths.begin(), scan.imagePaths.end());
	if (scan.imageCount == 0) {
		scan.warnings.push_back("dataset contains no supported images");
	}
	return scan;
}

bool ofxGgmlDiffusionWriteTinyGanFixtureDataset(
	const std::string& datasetPath,
	int imageCount,
	std::string& error) {
	error.clear();
	if (datasetPath.empty()) {
		error = "datasetPath is required";
		return false;
	}
	if (imageCount < 1 || imageCount > 1024) {
		error = "imageCount must be between 1 and 1024";
		return false;
	}

	const std::filesystem::path root(datasetPath);
	std::error_code code;
	std::filesystem::create_directories(root, code);
	if (code) {
		error = "could not create fixture dataset directory";
		return false;
	}

	for (int index = 0; index < imageCount; ++index) {
		std::ostringstream name;
		name << "fixture-" << std::setw(3) << std::setfill('0') << index << ".ppm";
		const auto path = root / name.str();
		std::ofstream file(path, std::ios::out | std::ios::trunc);
		if (!file) {
			error = "could not write fixture image: " + path.string();
			return false;
		}

		file << "P3\n64 64\n255\n";
		for (int y = 0; y < 64; ++y) {
			for (int x = 0; x < 64; ++x) {
				file << fixtureChannelValue(x, y, index, 0) << " "
					<< fixtureChannelValue(x, y, index, 1) << " "
					<< fixtureChannelValue(x, y, index, 2);
				if (x < 63) {
					file << " ";
				}
			}
			file << "\n";
		}
	}

	return true;
}

ofxGgmlDiffusionTinyGanDiscriminatorResult ofxGgmlDiffusionRunTinyGanDiscriminatorForward(
	const ofxGgmlDiffusionTinyGanImageSample& sample,
	int hiddenSize) {
	ofxGgmlDiffusionTinyGanDiscriminatorResult result;
	result.hiddenSize = hiddenSize;
	if (!sample.isAllocated()) {
		result.error = "sample is not allocated";
		return result;
	}
	if (sample.normalizedPixels.size() != sample.pixels.size()) {
		result.error = "sample normalized pixel count does not match byte pixel count";
		return result;
	}
	if (sample.width != 64 || sample.height != 64 || sample.channels != 3) {
		result.error = "tiny discriminator currently supports 64x64 RGB samples only";
		return result;
	}
	if (hiddenSize < 8 || hiddenSize > 512) {
		result.error = "hiddenSize must be between 8 and 512";
		return result;
	}

	float logit = -0.04f;
	const auto& pixels = sample.normalizedPixels;
	for (int hidden = 0; hidden < hiddenSize; ++hidden) {
		float activation = static_cast<float>((hidden % 7) - 3) * 0.015f;
		const std::size_t stride = static_cast<std::size_t>((hidden % 5) + 3);
		for (std::size_t i = static_cast<std::size_t>(hidden % 3); i < pixels.size(); i += stride) {
			activation += pixels[i] * fixedTinyDiscriminatorWeight(i, hidden);
		}
		activation = std::tanh(activation / static_cast<float>(pixels.size() / stride));
		const float outputWeight = static_cast<float>((hidden * 13) % 19 - 9) / static_cast<float>(hiddenSize);
		logit += activation * outputWeight;
	}

	result.success = true;
	result.logit = logit;
	result.probability = sigmoid(logit);
	return result;
}

float ofxGgmlDiffusionTinyGanBinaryCrossEntropy(
	float probability,
	bool targetReal) {
	const float clamped = clampProbability(probability);
	return targetReal ? -std::log(clamped) : -std::log(1.0f - clamped);
}

ofxGgmlDiffusionTinyGanLossPair ofxGgmlDiffusionComputeTinyGanLossPair(
	float realProbability,
	float fakeProbability,
	float generatorFakeProbability) {
	ofxGgmlDiffusionTinyGanLossPair loss;
	loss.realLoss = ofxGgmlDiffusionTinyGanBinaryCrossEntropy(realProbability, true);
	loss.fakeLoss = ofxGgmlDiffusionTinyGanBinaryCrossEntropy(fakeProbability, false);
	loss.discriminatorLoss = 0.5f * (loss.realLoss + loss.fakeLoss);
	loss.generatorLoss = ofxGgmlDiffusionTinyGanBinaryCrossEntropy(generatorFakeProbability, true);
	return loss;
}

ofxGgmlDiffusionTinyGanWeightUpdatePreview ofxGgmlDiffusionPreviewTinyGanWeightUpdate(
	const ofxGgmlDiffusionTinyGanLossPair& loss,
	float learningRate) {
	ofxGgmlDiffusionTinyGanWeightUpdatePreview preview;
	preview.initialDiscriminatorWeight = 0.125f;
	preview.initialGeneratorWeight = -0.075f;
	preview.learningRate = isFinitePositive(learningRate) ? learningRate : 0.001f;
	preview.discriminatorGradient = loss.fakeLoss - loss.realLoss;
	preview.generatorGradient = -loss.generatorLoss;
	preview.updatedDiscriminatorWeight =
		preview.initialDiscriminatorWeight - (preview.learningRate * preview.discriminatorGradient);
	preview.updatedGeneratorWeight =
		preview.initialGeneratorWeight - (preview.learningRate * preview.generatorGradient);
	return preview;
}

std::vector<float> ofxGgmlDiffusionNormalizeTinyGanImage(
	const ofxGgmlDiffusionTinyGanImageSample& sample) {
	std::vector<float> normalized;
	normalized.reserve(sample.pixels.size());
	for (const auto value : sample.pixels) {
		normalized.push_back((static_cast<float>(value) / 127.5f) - 1.0f);
	}
	return normalized;
}

bool ofxGgmlDiffusionLoadTinyGanPpmImage(
	const std::string& imagePath,
	ofxGgmlDiffusionTinyGanImageSample& sample,
	std::string& error) {
	error.clear();
	sample = ofxGgmlDiffusionTinyGanImageSample{};
	if (imagePath.empty()) {
		error = "imagePath is required";
		return false;
	}

	std::ifstream file(imagePath);
	if (!file) {
		error = "could not open PPM image: " + imagePath;
		return false;
	}

	std::string magic;
	if (!readPpmToken(file, magic) || magic != "P3") {
		error = "only ASCII P3 PPM images are supported";
		return false;
	}

	int width = 0;
	int height = 0;
	int maxValue = 0;
	if (!parsePpmInt(file, width) ||
		!parsePpmInt(file, height) ||
		!parsePpmInt(file, maxValue)) {
		error = "invalid PPM header";
		return false;
	}
	if (width <= 0 || height <= 0) {
		error = "PPM dimensions must be positive";
		return false;
	}
	if (maxValue <= 0 || maxValue > 255) {
		error = "PPM max value must be between 1 and 255";
		return false;
	}

	const int channels = 3;
	const std::size_t valueCount = static_cast<std::size_t>(width) *
		static_cast<std::size_t>(height) *
		static_cast<std::size_t>(channels);
	std::vector<std::uint8_t> pixels;
	pixels.reserve(valueCount);
	for (std::size_t i = 0; i < valueCount; ++i) {
		int value = 0;
		if (!parsePpmInt(file, value)) {
			error = "PPM ended before all pixels were read";
			return false;
		}
		if (value < 0 || value > maxValue) {
			error = "PPM pixel value out of range";
			return false;
		}
		const float scaled = (static_cast<float>(value) / static_cast<float>(maxValue)) * 255.0f;
		pixels.push_back(static_cast<std::uint8_t>(std::round(scaled)));
	}

	sample.width = width;
	sample.height = height;
	sample.channels = channels;
	sample.pixels = std::move(pixels);
	sample.normalizedPixels = ofxGgmlDiffusionNormalizeTinyGanImage(sample);
	return true;
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
			const float realProbability = 0.50f + (0.18f * progress);
			const float fakeProbability = 0.50f - (0.12f * progress);
			const float generatorFakeProbability = 0.50f + (0.07f * progress);
			const auto loss = ofxGgmlDiffusionComputeTinyGanLossPair(
				realProbability,
				fakeProbability,
				generatorFakeProbability);
			ofxGgmlDiffusionTinyGanTrainingStep step;
			step.epoch = epoch + 1;
			step.batch = batch + 1;
			step.realProbability = realProbability;
			step.fakeProbability = fakeProbability;
			step.discriminatorLoss = loss.discriminatorLoss;
			step.generatorLoss = loss.generatorLoss;
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
	result.dataset = ofxGgmlDiffusionScanTinyGanDataset(settings.datasetPath);
	for (const auto& warning : result.dataset.warnings) {
		result.warnings.push_back(warning);
	}
	result.steps = ofxGgmlDiffusionBuildTinyGanTrainingDryRunSteps(settings);
	if (!result.steps.empty()) {
		result.finalDiscriminatorLoss = result.steps.back().discriminatorLoss;
		result.finalGeneratorLoss = result.steps.back().generatorLoss;
		const auto finalLoss = ofxGgmlDiffusionComputeTinyGanLossPair(
			result.steps.back().realProbability,
			result.steps.back().fakeProbability,
			result.steps.back().fakeProbability);
		result.updatePreview = ofxGgmlDiffusionPreviewTinyGanWeightUpdate(finalLoss, settings.learningRate);
	}

	std::ostringstream text;
	text << "tiny GAN training dry-run\n";
	text << "dataset: " << (settings.datasetPath.empty() ? "(synthetic placeholder)" : settings.datasetPath) << "\n";
	text << "dataset exists: " << (result.dataset.exists ? "yes" : "no") << "\n";
	text << "dataset images: " << result.dataset.imageCount << "\n";
	text << "dataset unsupported files: " << result.dataset.unsupportedFileCount << "\n";
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
	text << "preview discriminator weight: " << result.updatePreview.updatedDiscriminatorWeight << "\n";
	text << "preview generator weight: " << result.updatePreview.updatedGeneratorWeight << "\n";
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
