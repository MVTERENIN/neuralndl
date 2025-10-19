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

// Load dataset from current directory
async function loadData() {
    try {
        const dataStatus = document.getElementById('dataStatus');
        dataStatus.innerHTML = '<span class="loading">Loading data.csv from current directory...</span>';
        
        // Load CSV file from same directory
        const response = await fetch('data.csv');
        if (!response.ok) {
            throw new Error(`Failed to load data.csv: ${response.status} ${response.statusText}`);
        }
        
        const csvText = await response.text();
        console.log('CSV loaded successfully, first 500 chars:', csvText.substring(0, 500));
        
        rawData = parseCSV(csvText);
        console.log('Parsed data sample:', rawData.slice(0, 3));
        
        if (rawData.length === 0) {
            throw new Error('No data found in CSV file');
        }
        
        dataStatus.innerHTML = '<span style="color: green;">✓ Data loaded successfully</span>';
        displayDataInfo(rawData);
        displayDataPreview(rawData);
        
        // Auto-proceed to preprocessing
        await preprocessDataAutomatically();
        
    } catch (error) {
        const errorMessage = `Error loading data: ${error.message}. Make sure data.csv is in the same folder as index.html`;
        console.error(errorMessage);
        document.getElementById('dataStatus').innerHTML = `<span class="error">${errorMessage}</span>`;
    }
}

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
                    <p>Architecture: Input(26) → Dense(64, relu) → Dropout(0.4) → Dense(32, relu) → Dropout(0.3) → Dense(1, sigmoid)</p>
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

// Parse CSV text to array of objects
function parseCSV(csvText) {
    const lines = csvText.trim().split('\n');
    if (lines.length === 0) return [];
    
    // Handle headers - convert to lowercase and trim
    const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
    console.log('CSV Headers:', headers);
    
    const data = [];
    
    for (let i = 1; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue; // Skip empty lines
        
        const values = line.split(',').map(v => v.trim());
        
        // Create object with lowercase keys
        const obj = {};
        headers.forEach((header, index) => {
            obj[header] = values[index] || '';
        });
        
        data.push(obj);
    }
    
    return data;
}

// Display dataset information
function displayDataInfo(data) {
    const infoDiv = document.getElementById('dataInfo');
    const missingCounts = calculateMissingValues(data);
    
    // Count class distribution for display
    let employableCount = 0;
    let lessEmployableCount = 0;
    
    data.forEach(row => {
        const targetValue = row[TARGET_COLUMN.toLowerCase()];
        if (targetValue) {
            const targetLower = targetValue.toString().toLowerCase();
            if (targetLower.includes('employable') && !targetLower.includes('less')) {
                employableCount++;
            } else if (targetLower.includes('less')) {
                lessEmployableCount++;
            }
        }
    });
    
    infoDiv.innerHTML = `
        <h3>Dataset Information</h3>
        <p><strong>Shape:</strong> ${data.length} rows × ${Object.keys(data[0]).length} columns</p>
        <p><strong>Target Variable:</strong> ${TARGET_COLUMN.toLowerCase()}</p>
        <p><strong>Class Distribution:</strong> ${employableCount} Employable, ${lessEmployableCount} LessEmployable</p>
        <p><strong>Base Features:</strong> ${FEATURE_COLUMNS.join(', ')}</p>
        <p><strong>Enhanced Features:</strong> Polynomial (squares and interactions)</p>
        <h4>Missing Values:</h4>
        <table>
            <tr><th>Column</th><th>Missing %</th></tr>
            ${Object.entries(missingCounts).map(([col, count]) => 
                `<tr><td>${col}</td><td>${((count/data.length)*100).toFixed(2)}%</td></tr>`
            ).join('')}
        </table>
    `;
}

// Calculate missing values percentage
function calculateMissingValues(data) {
    const missing = {};
    if (data.length === 0) return missing;
    
    Object.keys(data[0]).forEach(col => {
        missing[col] = data.filter(row => !row[col] || row[col] === '').length;
    });
    return missing;
}

// Display data preview
function displayDataPreview(data) {
    const previewDiv = document.getElementById('dataPreview');
    const previewData = data.slice(0, 5);
    
    if (previewData.length === 0) {
        previewDiv.innerHTML = '<p>No data to display</p>';
        return;
    }
    
    previewDiv.innerHTML = `
        <h3>Data Preview (first 5 rows)</h3>
        <table>
            <tr>${Object.keys(data[0]).map(col => `<th>${col}</th>`).join('')}</tr>
            ${previewData.map(row => 
                `<tr>${Object.values(row).map(val => `<td>${val}</td>`).join('')}</tr>`
            ).join('')}
        </table>
    `;
}

// Automatically preprocess data after loading
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
                inputShape: [26], // 8 original + 5 basic + 8 squares + 5 interactions = 26 features
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
    
    return history;
}

// Create data visualization
function createDataVisualization(classDistribution) {
    const data = [
        { index: 'Employable (1)', value: classDistribution.employable },
        { index: 'LessEmployable (0)', value: classDistribution.lessEmployable }
    ];

    const surface = { name: 'Class Distribution', tab: 'Data Analysis' };
    tfvis.render.barchart(surface, data, {
        xLabel: 'Class',
        yLabel: 'Count',
        width: 400,
        height: 300
    });
}

// Predict for single input
function predictSingle() {
    if (!model) {
        alert('Please create and train the model first!');
        return;
    }
    
    try {
        // Get values from form
        const baseFeatures = [
            parseFloat(document.getElementById('generalAppearance').value),
            parseFloat(document.getElementById('mannerSpeaking').value),
            parseFloat(document.getElementById('physicalCondition').value),
            parseFloat(document.getElementById('mentalAlertness').value),
            parseFloat(document.getElementById('selfConfidence').value),
            parseFloat(document.getElementById('abilityPresentIdeas').value),
            parseFloat(document.getElementById('communicationSkills').value),
            parseFloat(document.getElementById('performanceRating').value)
        ];
        
        // Validate inputs
        for (let i = 0; i < baseFeatures.length; i++) {
            if (isNaN(baseFeatures[i]) || baseFeatures[i] < 2 || baseFeatures[i] > 5) {
                alert(`Please enter valid values between 2 and 5 for all fields. Invalid value at: ${FEATURE_COLUMNS[i]}`);
                return;
            }
        }
        
        // Create enhanced features (same as during training)
        const enhancedFeatures = createPolynomialFeatures([baseFeatures])[0];
        
        // Create tensor and predict
        const inputTensor = tf.tensor2d([enhancedFeatures]);
        const prediction = model.predict(inputTensor);
        const probability = prediction.dataSync()[0];
        prediction.dispose();
        inputTensor.dispose();
        
        // Get current threshold
        const threshold = parseFloat(document.getElementById('thresholdSlider').value);
        const predictedClass = probability >= threshold ? 'Employable' : 'LessEmployable';
        
        // Display results
        const resultDiv = document.getElementById('singlePredictionResult');
        const confidence = (probability * 100).toFixed(2);
        const confidencePercent = Math.min(100, Math.max(0, Math.abs(probability - threshold) * 200));
        
        resultDiv.innerHTML = `
            <div class="prediction-result ${probability >= threshold ? 'result-employable' : 'result-lessemployable'}">
                <h4 style="margin-top: 0;">Prediction Result (Balanced Model)</h4>
                <p><strong>Predicted Class:</strong> ${predictedClass}</p>
                <p><strong>Probability:</strong> ${confidence}%</p>
                <p><strong>Threshold:</strong> ${(threshold * 100).toFixed(2)}%</p>
                
                <div class="confidence-bar">
                    <div class="confidence-fill ${probability >= threshold ? 'confidence-employable' : 'confidence-lessemployable'}" 
                         style="width: ${confidencePercent}%"></div>
                </div>
                
                <p><strong>Confidence:</strong> 
                    <span style="color: ${probability >= threshold ? '#28a745' : '#dc3545'}; font-weight: bold;">
                        ${Math.abs((probability - threshold) * 100).toFixed(2)}% ${probability >= threshold ? 'above' : 'below'} threshold
                    </span>
                </p>
                
                <div style="margin-top: 15px; padding: 10px; background: #f8f9fa; border-radius: 5px;">
                    <small><strong>Feature Analysis:</strong><br>
                    Total Score: ${baseFeatures.reduce((a, b) => a + b, 0)}<br>
                    Average Score: ${(baseFeatures.reduce((a, b) => a + b, 0) / baseFeatures.length).toFixed(2)}<br>
                    Min Score: ${Math.min(...baseFeatures)} | Max Score: ${Math.max(...baseFeatures)}
                    </small>
                </div>
            </div>
        `;
        
    } catch (error) {
        alert(`Error making prediction: ${error.message}`);
    }
}

// Evaluate model with given threshold
async function evaluateModel(threshold) {
    try {
        // Make predictions on validation set
        const predictions = model.predict(validationData);
        const probs = await predictions.data();
        predictions.dispose();
        
        // Convert probabilities to binary predictions using threshold
        const binaryPreds = probs.map(p => p >= threshold ? 1 : 0);
        const trueLabels = await validationLabels.data();
        
        // Calculate metrics
        const metrics = calculateMetrics(trueLabels, binaryPreds);
        
        // Update ROC curve if not already calculated
        if (!validationPredictions) {
            validationPredictions = probs;
            plotROCCurve(trueLabels, probs);
        }
        
        // Display metrics
        displayMetrics(metrics, threshold);
        
    } catch (error) {
        console.error('Error evaluating model:', error);
    }
}

// Calculate classification metrics
function calculateMetrics(trueLabels, predictions) {
    let tp = 0, fp = 0, tn = 0, fn = 0;
    
    for (let i = 0; i < trueLabels.length; i++) {
        if (trueLabels[i] === 1 && predictions[i] === 1) tp++;
        else if (trueLabels[i] === 0 && predictions[i] === 1) fp++;
        else if (trueLabels[i] === 0 && predictions[i] === 0) tn++;
        else if (trueLabels[i] === 1 && predictions[i] === 0) fn++;
    }
    
    const accuracy = (tp + tn) / (tp + fp + tn + fn);
    const precision = tp / (tp + fp) || 0;
    const recall = tp / (tp + fn) || 0;
    const f1 = 2 * (precision * recall) / (precision + recall) || 0;
    
    return { tp, fp, tn, fn, accuracy, precision, recall, f1 };
}

// Display metrics and confusion matrix
function displayMetrics(metrics, threshold) {
    const confusionDiv = document.getElementById('confusionMatrix');
    const metricsDiv = document.getElementById('metricsDisplay');
    
    confusionDiv.innerHTML = `
        <h4>Confusion Matrix (Threshold: ${threshold.toFixed(2)})</h4>
        <table>
            <tr><th></th><th>Predicted LessEmployable (0)</th><th>Predicted Employable (1)</th></tr>
            <tr><th>Actual LessEmployable (0)</th><td>${metrics.tn}</td><td>${metrics.fp}</td></tr>
            <tr><th>Actual Employable (1)</th><td>${metrics.fn}</td><td>${metrics.tp}</td></tr>
        </table>
    `;
    
    metricsDiv.innerHTML = `
        <h4>Performance Metrics (Balanced Model)</h4>
        <p><strong>Accuracy:</strong> ${(metrics.accuracy * 100).toFixed(2)}%</p>
        <p><strong>Precision:</strong> ${(metrics.precision * 100).toFixed(2)}%</p>
        <p><strong>Recall:</strong> ${(metrics.recall * 100).toFixed(2)}%</p>
        <p><strong>F1-Score:</strong> ${(metrics.f1 * 100).toFixed(2)}%</p>
    `;
}

// Plot ROC curve and calculate AUC
function plotROCCurve(trueLabels, probabilities) {
    // Calculate ROC curve points
    const thresholds = Array.from({ length: 100 }, (_, i) => i / 100);
    const rocPoints = thresholds.map(threshold => {
        const binaryPreds = probabilities.map(p => p >= threshold ? 1 : 0);
        const metrics = calculateMetrics(trueLabels, binaryPreds);
        return {
            fpr: metrics.fp / (metrics.fp + metrics.tn) || 0,
            tpr: metrics.tp / (metrics.tp + metrics.fn) || 0,
            threshold
        };
    });
    
    // Calculate AUC (Area Under Curve)
    let auc = 0;
    for (let i = 1; i < rocPoints.length; i++) {
        auc += (rocPoints[i].fpr - rocPoints[i-1].fpr) * 
               (rocPoints[i].tpr + rocPoints[i-1].tpr) / 2;
    }
    
    // Plot ROC curve
    const rocData = {
        values: rocPoints.map(point => ({ x: point.fpr, y: point.tpr })),
        series: ['ROC Curve']
    };
    
    const surface = { name: 'ROC Curve', tab: 'Evaluation' };
    tfvis.render.linechart(surface, rocData, {
        xLabel: 'False Positive Rate',
        yLabel: 'True Positive Rate',
        width: 400,
        height: 400
    });
    
    // Display AUC
    document.getElementById('rocCurve').innerHTML += `
        <p><strong>AUC:</strong> ${auc.toFixed(3)}</p>
        <p><strong>Model:</strong> Balanced (64-32 neurons with class weights)</p>
    `;
}
