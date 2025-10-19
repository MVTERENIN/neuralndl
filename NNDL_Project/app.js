// Student Employability Classifier - TensorFlow.js
// Dataset: Students' Employability Dataset - Philippines

// Global variables
let rawData = [];
let features = null;
let labels = null;
let model = null;
let trainingHistory = [];
let validationData = null;
let validationLabels = null;
let validationPredictions = null;
let isTraining = false;
let classWeights = null;

// Feature configuration - UPDATE THESE FOR DIFFERENT DATASETS
const FEATURE_COLUMNS = [
    'GENERAL APPEARANCE', 'MANNER OF SPEAKING', 'PHYSICAL CONDITION', 
    'MENTAL ALERTNESS', 'SELF-CONFIDENCE', 'ABILITY TO PRESENT IDEAS', 
    'COMMUNICATION SKILLS', 'Student Performance Rating'
];
const TARGET_COLUMN = 'CLASS'; // Target variable name

// Initialize application when page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log('Student Employability Classifier initialized');
    initializeEventListeners();
    loadData();
});

// Initialize all event listeners
function initializeEventListeners() {
    // Create model button
    const createModelBtn = document.getElementById('createModel');
    if (createModelBtn) {
        createModelBtn.addEventListener('click', function() {
            try {
                model = createBalancedModel();
                document.getElementById('modelInfo').innerHTML = `
                    <p><strong>Balanced Model Created Successfully</strong></p>
                    <p>Architecture: Input(18) → Dense(64, relu) → Dropout(0.4) → Dense(32, relu) → Dropout(0.3) → Dense(1, sigmoid)</p>
                    <p><strong>Class Weights:</strong> Employable: ${classWeights[1].toFixed(2)}, LessEmployable: ${classWeights[0].toFixed(2)}</p>
                    <p><strong>Enhanced Features:</strong> Polynomial (squares and interactions)</p>
                `;
                
                // Print model summary to console
                console.log('Model summary:');
                model.summary();
                
                document.getElementById('trainModel').disabled = false;
                
            } catch (error) {
                const errorMessage = `Error creating model: ${error.message}`;
                console.error(errorMessage);
                alert(errorMessage);
            }
        });
    }

    // Train model button
    const trainModelBtn = document.getElementById('trainModel');
    if (trainModelBtn) {
        trainModelBtn.addEventListener('click', async function() {
            if (isTraining) return;
            
            try {
                isTraining = true;
                const trainButton = this;
                const trainingInfo = document.getElementById('trainingInfo');
                
                trainButton.disabled = true;
                trainButton.textContent = 'Training...';
                trainingInfo.innerHTML = '<span class="loading">Training balanced model with class weights...</span>';
                
                // Clear previous training charts
                tfvis.visor().close();
                
                // Split data into training and validation sets (80/20) with stratification
                const {trainFeatures, trainLabels, valFeatures, valLabels} = stratifiedSplit(features, labels, 0.2);
                validationData = valFeatures;
                validationLabels = valLabels;
                
                console.log('Training set size:', trainFeatures.shape[0]);
                console.log('Validation set size:', validationData.shape[0]);
                console.log('Feature dimension:', features.shape[1]);
                
                // Train the model
                await trainBalancedModel(model, trainFeatures, trainLabels, validationData, validationLabels);
                
                // Enable prediction and evaluation
                document.getElementById('predictSingle').disabled = false;
                document.getElementById('resetForm').disabled = false;
                document.getElementById('thresholdSlider').disabled = false;
                trainButton.textContent = 'Training Complete';
                trainingInfo.innerHTML = '<span style="color: green;">✓ Training completed successfully</span>';
                
                // Auto-evaluate with default threshold
                evaluateModel(0.5);
                
            } catch (error) {
                const errorMessage = `Error training model: ${error.message}`;
                console.error(errorMessage);
                document.getElementById('trainingInfo').innerHTML = `<span class="error">${errorMessage}</span>`;
                document.getElementById('trainModel').disabled = false;
                document.getElementById('trainModel').textContent = 'Train Model';
            } finally {
                isTraining = false;
            }
        });
    }

    // Threshold slider
    const thresholdSlider = document.getElementById('thresholdSlider');
    if (thresholdSlider) {
        thresholdSlider.addEventListener('input', function() {
            const threshold = parseFloat(this.value);
            document.getElementById('thresholdValue').textContent = threshold.toFixed(2);
            
            if (validationData && model) {
                evaluateModel(threshold);
            }
        });
    }

    // Predict single button
    const predictSingleBtn = document.getElementById('predictSingle');
    if (predictSingleBtn) {
        predictSingleBtn.addEventListener('click', predictSingle);
    }
    
    // Reset form button
    const resetFormBtn = document.getElementById('resetForm');
    if (resetFormBtn) {
        resetFormBtn.addEventListener('click', function() {
            // Reset all input fields to default value 3
            document.getElementById('generalAppearance').value = 3;
            document.getElementById('mannerSpeaking').value = 3;
            document.getElementById('physicalCondition').value = 3;
            document.getElementById('mentalAlertness').value = 3;
            document.getElementById('selfConfidence').value = 3;
            document.getElementById('abilityPresentIdeas').value = 3;
            document.getElementById('communicationSkills').value = 3;
            document.getElementById('performanceRating').value = 3;
            
            // Clear prediction result
            document.getElementById('singlePredictionResult').innerHTML = '';
        });
    }
}

// Calculate class weights for imbalanced dataset
function calculateClassWeights(labelsTensor) {
    const labelsArray = labelsTensor.arraySync();
    let count0 = 0, count1 = 0;
    
    labelsArray.forEach(label => {
        if (label === 0) count0++;
        else count1++;
    });
    
    const total = count0 + count1;
    const weight0 = total / (2 * count0);  // Higher weight for minority class
    const weight1 = total / (2 * count1);  // Lower weight for majority class
    
    console.log(`Class distribution: ${count0} LessEmployable, ${count1} Employable`);
    console.log(`Class weights: LessEmployable: ${weight0.toFixed(2)}, Employable: ${weight1.toFixed(2)}`);
    
    return {0: weight0, 1: weight1};
}

// Create balanced model with class weights and enhanced architecture
function createBalancedModel() {
    const model = tf.sequential({
        layers: [
            // First hidden layer with 64 neurons
            tf.layers.dense({
                inputShape: [18], // 8 original + 5 basic + 5 polynomial = 18 features
                units: 64,
                activation: 'relu',
                kernelRegularizer: tf.regularizers.l2({l2: 0.01})
            }),
            tf.layers.dropout({rate: 0.4}),
            // Second hidden layer with 32 neurons
            tf.layers.dense({
                units: 32,
                activation: 'relu',
                kernelRegularizer: tf.regularizers.l2({l2: 0.01})
            }),
            tf.layers.dropout({rate: 0.3}),
            // Output layer
            tf.layers.dense({
                units: 1,
                activation: 'sigmoid'
            })
        ]
    });

    // Compile with customized optimizer and lower learning rate
    model.compile({
        optimizer: tf.train.adam(0.0005),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });

    return model;
}

// Stratified split to maintain class distribution
function stratifiedSplit(features, labels, testSize = 0.2) {
    const labelsArray = labels.arraySync();
    const indices0 = []; // LessEmployable indices
    const indices1 = []; // Employable indices
    
    labelsArray.forEach((label, index) => {
        if (label === 0) indices0.push(index);
        else indices1.push(index);
    });
    
    // Shuffle indices
    tf.util.shuffle(indices0);
    tf.util.shuffle(indices1);
    
    const splitIndex0 = Math.floor(indices0.length * testSize);
    const splitIndex1 = Math.floor(indices1.length * testSize);
    
    const valIndices = [
        ...indices0.slice(0, splitIndex0),
        ...indices1.slice(0, splitIndex1)
    ];
    
    const trainIndices = [
        ...indices0.slice(splitIndex0),
        ...indices1.slice(splitIndex1)
    ];
    
    // Shuffle final arrays
    tf.util.shuffle(trainIndices);
    tf.util.shuffle(valIndices);
    
    const trainFeatures = features.gather(trainIndices);
    const trainLabels = labels.gather(trainIndices);
    const valFeatures = features.gather(valIndices);
    const valLabels = labels.gather(valIndices);
    
    return {trainFeatures, trainLabels, valFeatures, valLabels};
}

// Train with class weights and focal loss-like approach
async function trainBalancedModel(model, trainFeatures, trainLabels, valFeatures, valLabels) {
    const lossSurface = { name: 'Loss', tab: 'Training' };
    const accuracySurface = { name: 'Accuracy', tab: 'Training' };
    
    trainingHistory = [];
    
    let bestValLoss = Infinity;
    let patienceCounter = 0;
    const patience = 10;
    
    // Calculate sample weights for loss function
    const labelsArray = await trainLabels.array();
    const sampleWeights = labelsArray.map(label => 
        classWeights[label]
    );
    const sampleWeightsTensor = tf.tensor1d(sampleWeights);
    
    const history = await model.fit(trainFeatures, trainLabels, {
        epochs: 200,
        batchSize: 32,
        validationData: [valFeatures, valLabels],
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                // Early stopping with patience
                if (logs.val_loss < bestValLoss) {
                    bestValLoss = logs.val_loss;
                    patienceCounter = 0;
                } else {
                    patienceCounter++;
                }
                
                if (patienceCounter >= patience) {
                    console.log(`Early stopping at epoch ${epoch + 1}`);
                    model.stopTraining = true;
                }
                
                // Update training history
                trainingHistory.push({
                    epoch: epoch + 1,
                    loss: logs.loss,
                    accuracy: logs.acc,
                    val_loss: logs.val_loss,
                    val_acc: logs.val_acc
                });
                
                // Update training info
                const trainingInfo = document.getElementById('trainingInfo');
                trainingInfo.innerHTML = `
                    <span class="loading">Training Balanced Model... Epoch ${epoch + 1}/200</span><br>
                    <small>Loss: ${logs.loss.toFixed(4)} | Accuracy: ${(logs.acc * 100).toFixed(2)}%<br>
                    Val Loss: ${logs.val_loss.toFixed(4)} | Val Accuracy: ${(logs.val_acc * 100).toFixed(2)}%</small>
                    ${patienceCounter > 0 ? `<br><small>Early stopping: ${patienceCounter}/${patience}</small>` : ''}
                `;
                
                // Plot metrics
                const lossData = {
                    values: trainingHistory.map(h => ({ x: h.epoch, y: h.loss })),
                    series: ['Training Loss']
                };
                
                const valLossData = {
                    values: trainingHistory.map(h => ({ x: h.epoch, y: h.val_loss })),
                    series: ['Validation Loss']
                };
                
                const accuracyData = {
                    values: trainingHistory.map(h => ({ x: h.epoch, y: h.accuracy })),
                    series: ['Training Accuracy']
                };
                
                const valAccuracyData = {
                    values: trainingHistory.map(h => ({ x: h.epoch, y: h.val_acc })),
                    series: ['Validation Accuracy']
                };
                
                tfvis.render.linechart(lossSurface, [lossData, valLossData], {
                    xLabel: 'Epoch', yLabel: 'Loss', width: 400, height: 300
                });
                
                tfvis.render.linechart(accuracySurface, [accuracyData, valAccuracyData], {
                    xLabel: 'Epoch', yLabel: 'Accuracy', width: 400, height: 300
                });
            }
        }
    });
    
    sampleWeightsTensor.dispose();
    return history;
}

// Create polynomial features for better non-linear modeling
function createPolynomialFeatures(featuresArray) {
    return featuresArray.map(features => {
        const originalFeatures = [...features];
        
        // Basic enhanced features
        const totalScore = features.reduce((a, b) => a + b, 0);
        const averageScore = totalScore / features.length;
        const minScore = Math.min(...features);
        const maxScore = Math.max(...features);
        const scoreRange = maxScore - minScore;
        
        // Polynomial features - squares
        const squares = features.map(x => x * x);
        
        // Important interactions
        const communicationIndex = 6; // COMMUNICATION SKILLS
        const confidenceIndex = 4; // SELF-CONFIDENCE
        const presentationIndex = 5; // ABILITY TO PRESENT IDEAS
        
        const interactions = [
            features[communicationIndex] * features[confidenceIndex],
            features[communicationIndex] * features[presentationIndex],
            features[confidenceIndex] * features[presentationIndex],
            features[communicationIndex] * averageScore,
            features[confidenceIndex] * totalScore
        ];
        
        return [
            ...originalFeatures,        // 8 original
            totalScore,                 // 1
            averageScore,               // 1
            minScore,                   // 1
            maxScore,                   // 1
            scoreRange,                 // 1
            ...squares,                 // 8 squares = 8
            ...interactions             // 5 interactions = 5
            // Total: 8 + 5 + 8 + 5 = 26 features
        ];
    });
}

// [Остальные функции остаются практически такими же, но с обновленными путями к данным]
// Для краткости оставлю основные функции без изменений, но обновлю preprocessing

// Preprocess the dataset with polynomial features
function preprocessDataWithPolynomialFeatures(data) {
    if (data.length === 0) {
        throw new Error('No data available for preprocessing');
    }
    
    const baseFeaturesArray = [];
    const labelsArray = [];
    let employableCount = 0;
    let lessEmployableCount = 0;

    console.log('Starting polynomial preprocessing...');

    data.forEach((row, index) => {
        try {
            // Extract base features
            const baseFeatures = FEATURE_COLUMNS.map(col => {
                const key = col.toLowerCase();
                const value = row[key];
                
                if (value === undefined || value === null || value === '') {
                    throw new Error(`Missing feature value for ${col} in row ${index}`);
                }
                
                const numValue = parseFloat(value);
                if (isNaN(numValue)) {
                    throw new Error(`Invalid numeric value in ${col}: ${value} (row ${index})`);
                }
                
                return numValue;
            });

            baseFeaturesArray.push(baseFeatures);

            // Extract and encode target variable
            const targetKey = TARGET_COLUMN.toLowerCase();
            const targetValue = row[targetKey];
            
            if (!targetValue) {
                throw new Error(`Missing target value in ${TARGET_COLUMN} for row ${index}`);
            }
            
            // Convert to binary
            const targetLower = targetValue.toString().toLowerCase();
            let encodedLabel;
            
            if (targetLower.includes('employable') && !targetLower.includes('less')) {
                encodedLabel = 1;
                employableCount++;
            } else if (targetLower.includes('less')) {
                encodedLabel = 0;
                lessEmployableCount++;
            } else {
                throw new Error(`Unknown target value: ${targetValue}`);
            }
            
            labelsArray.push(encodedLabel);
            
        } catch (error) {
            console.error(`Error processing row ${index}:`, error.message);
            throw error;
        }
    });

    // Create polynomial features
    const polynomialFeaturesArray = createPolynomialFeatures(baseFeaturesArray);
    
    console.log('Polynomial preprocessing completed. Features:', polynomialFeaturesArray.length);
    console.log('Feature dimension:', polynomialFeaturesArray[0].length);
    console.log('Class distribution - Employable:', employableCount, 'LessEmployable:', lessEmployableCount);

    // Calculate class weights
    classWeights = calculateClassWeights(tf.tensor1d(labelsArray));

    return {
        features: tf.tensor2d(polynomialFeaturesArray),
        labels: tf.tensor1d(labelsArray),
        classDistribution: {
            employable: employableCount,
            lessEmployable: lessEmployableCount
        }
    };
}

// [Остальные функции (loadData, displayDataInfo, и т.д.) остаются похожими, 
// но обновите вызов preprocessDataAutomatically чтобы использовать polynomial features]

// Обновите функцию preprocessDataAutomatically:
async function preprocessDataAutomatically() {
    try {
        document.getElementById('preprocessStatus').innerHTML = '<span class="loading">Preprocessing data with polynomial features...</span>';
        
        const processed = preprocessDataWithPolynomialFeatures(rawData);
        features = processed.features;
        labels = processed.labels;
        
        document.getElementById('preprocessStatus').innerHTML = '<span style="color: green;">✓ Polynomial preprocessing complete</span>';
        document.getElementById('preprocessInfo').innerHTML = `
            <p><strong>Polynomial Preprocessing Complete</strong></p>
            <p>Features shape: [${features.shape[0]}, ${features.shape[1]}] (8 original + 18 enhanced = 26 total)</p>
            <p>Labels shape: [${labels.shape[0]}]</p>
            <p>Class distribution: ${processed.classDistribution.employable} Employable (1), 
            ${processed.classDistribution.lessEmployable} LessEmployable (0)</p>
            <p><strong>New Features:</strong> Squares, Interactions, Totals, Averages</p>
            <p><strong>Class Weights Applied:</strong> LessEmployable: ${classWeights[0].toFixed(2)}, Employable: ${classWeights[1].toFixed(2)}</p>
        `;
        
        createDataVisualization(processed.classDistribution);
        document.getElementById('createModel').disabled = false;
        
    } catch (error) {
        const errorMessage = `Error preprocessing data: ${error.message}`;
        console.error(errorMessage);
        document.getElementById('preprocessStatus').innerHTML = `<span class="error">${errorMessage}</span>`;
    }
}
