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
                model = createImprovedModel();
                document.getElementById('modelInfo').innerHTML = `
                    <p><strong>Improved Model Created Successfully</strong></p>
                    <p>Architecture: Input(${FEATURE_COLUMNS.length + 5}) → Dense(32, relu) → Dense(16, relu) → Dense(1, sigmoid)</p>
                    <p><strong>New Features:</strong> Total Score, Average Score, Min Score, Max Score, Score Range</p>
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
                trainingInfo.innerHTML = '<span class="loading">Training improved model... This may take a few moments.</span>';
                
                // Clear previous training charts
                tfvis.visor().close();
                
                // Split data into training and validation sets (80/20)
                const splitIndex = Math.floor(features.shape[0] * 0.8);
                
                const trainFeatures = features.slice(0, splitIndex);
                const trainLabels = labels.slice(0, splitIndex);
                validationData = features.slice(splitIndex);
                validationLabels = labels.slice(splitIndex);
                
                console.log('Training set size:', trainFeatures.shape[0]);
                console.log('Validation set size:', validationData.shape[0]);
                console.log('Feature dimension:', features.shape[1]);
                
                // Train the model
                await trainImprovedModel(model, trainFeatures, trainLabels, validationData, validationLabels);
                
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

// Create improved neural network model with more neurons and layers
function createImprovedModel() {
    const model = tf.sequential({
        layers: [
            // First hidden layer with 32 neurons and ReLU activation
            tf.layers.dense({
                inputShape: [FEATURE_COLUMNS.length + 5], // Original features + 5 new features
                units: 32,
                activation: 'relu',
                kernelRegularizer: tf.regularizers.l2({l2: 0.001}) // L2 regularization
            }),
            // Dropout layer to prevent overfitting
            tf.layers.dropout({rate: 0.3}),
            // Second hidden layer with 16 neurons and ReLU activation
            tf.layers.dense({
                units: 16,
                activation: 'relu',
                kernelRegularizer: tf.regularizers.l2({l2: 0.001}) // L2 regularization
            }),
            // Dropout layer
            tf.layers.dropout({rate: 0.3}),
            // Output layer with sigmoid activation for binary classification
            tf.layers.dense({
                units: 1,
                activation: 'sigmoid'
            })
        ]
    });

    // Compile the model with customized optimizer
    model.compile({
        optimizer: tf.train.adam(0.001), // Lower learning rate for better convergence
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });

    return model;
}

// Train the improved model with real-time metrics
async function trainImprovedModel(model, trainFeatures, trainLabels, valFeatures, valLabels) {
    // Create surfaces for metrics
    const lossSurface = { name: 'Loss', tab: 'Training' };
    const accuracySurface = { name: 'Accuracy', tab: 'Training' };
    
    // Clear previous data
    trainingHistory = [];
    
    // Early stopping callback
    let bestValLoss = Infinity;
    let patienceCounter = 0;
    const patience = 8; // Increased patience for more complex model
    
    const history = await model.fit(trainFeatures, trainLabels, {
        epochs: 100, // Increased epochs for better training
        batchSize: 16, // Smaller batch size for better generalization
        validationData: [valFeatures, valLabels],
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                // Early stopping logic
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
                
                // Update training info with current metrics
                const trainingInfo = document.getElementById('trainingInfo');
                trainingInfo.innerHTML = `
                    <span class="loading">Training Improved Model... Epoch ${epoch + 1}/100</span><br>
                    <small>Loss: ${logs.loss.toFixed(4)} | Accuracy: ${(logs.acc * 100).toFixed(2)}% | Val Loss: ${logs.val_loss.toFixed(4)} | Val Accuracy: ${(logs.val_acc * 100).toFixed(2)}%</small>
                    ${patienceCounter > 0 ? `<br><small>Early stopping counter: ${patienceCounter}/${patience}</small>` : ''}
                `;
                
                // Plot metrics in real-time
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
                
                // Render charts
                tfvis.render.linechart(lossSurface, [lossData, valLossData], {
                    xLabel: 'Epoch',
                    yLabel: 'Loss',
                    width: 400,
                    height: 300,
                    seriesColors: ['blue', 'red']
                });
                
                tfvis.render.linechart(accuracySurface, [accuracyData, valAccuracyData], {
                    xLabel: 'Epoch',
                    yLabel: 'Accuracy',
                    width: 400,
                    height: 300,
                    seriesColors: ['green', 'orange']
                });
            },
            onTrainEnd: () => {
                console.log('Training completed');
                // Display final training info
                const trainingInfo = document.getElementById('trainingInfo');
                trainingInfo.innerHTML += '<br><small>Training finished - final metrics displayed above</small>';
            }
        }
    });
    
    return history;
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
        const enhancedFeatures = createEnhancedFeatures([baseFeatures])[0];
        
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
                <h4 style="margin-top: 0;">Prediction Result (Improved Model)</h4>
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

// Create enhanced features with additional engineered features
function createEnhancedFeatures(featuresArray) {
    return featuresArray.map(features => {
        // Original features
        const originalFeatures = [...features];
        
        // New engineered features
        const totalScore = features.reduce((a, b) => a + b, 0);
        const averageScore = totalScore / features.length;
        const minScore = Math.min(...features);
        const maxScore = Math.max(...features);
        const scoreRange = maxScore - minScore;
        
        // Combine original and new features
        return [
            ...originalFeatures,
            totalScore,
            averageScore,
            minScore,
            maxScore,
            scoreRange
        ];
    });
}

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
        <p><strong>Enhanced Features:</strong> Total Score, Average Score, Min Score, Max Score, Score Range</p>
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
        document.getElementById('preprocessStatus').innerHTML = '<span class="loading">Preprocessing data with enhanced features...</span>';
        
        const processed = preprocessDataWithEnhancedFeatures(rawData);
        features = processed.features;
        labels = processed.labels;
        
        document.getElementById('preprocessStatus').innerHTML = '<span style="color: green;">✓ Preprocessing complete with enhanced features</span>';
        document.getElementById('preprocessInfo').innerHTML = `
            <p><strong>Enhanced Preprocessing Complete</strong></p>
            <p>Features shape: [${features.shape[0]}, ${features.shape[1]}] (original: 8 + new: 5 = 13 total)</p>
            <p>Labels shape: [${labels.shape[0]}]</p>
            <p>Class distribution: ${processed.classDistribution.employable} Employable (1), 
            ${processed.classDistribution.lessEmployable} LessEmployable (0)</p>
            <p><strong>New Features Added:</strong> Total Score, Average Score, Min Score, Max Score, Score Range</p>
        `;
        
        // Create visualization
        createDataVisualization(processed.classDistribution);
        
        // Enable model creation button
        document.getElementById('createModel').disabled = false;
        
    } catch (error) {
        const errorMessage = `Error preprocessing data: ${error.message}`;
        console.error(errorMessage);
        document.getElementById('preprocessStatus').innerHTML = `<span class="error">${errorMessage}</span>`;
    }
}

// Preprocess the dataset with enhanced features
function preprocessDataWithEnhancedFeatures(data) {
    if (data.length === 0) {
        throw new Error('No data available for preprocessing');
    }
    
    const featuresArray = [];
    const labelsArray = [];
    let employableCount = 0;
    let lessEmployableCount = 0;

    console.log('Starting enhanced preprocessing...');
    console.log('First row sample:', data[0]);

    data.forEach((row, index) => {
        try {
            // Extract base features (ignore case in column names)
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

            // Create enhanced features
            const enhancedFeatures = createEnhancedFeatures([baseFeatures])[0];
            featuresArray.push(enhancedFeatures);

            // Extract and encode target variable
            const targetKey = TARGET_COLUMN.toLowerCase();
            const targetValue = row[targetKey];
            
            if (!targetValue) {
                throw new Error(`Missing target value in ${TARGET_COLUMN} for row ${index}`);
            }
            
            // Convert to binary (1 for Employable, 0 for LessEmployable)
            const targetLower = targetValue.toString().toLowerCase();
            let encodedLabel;
            
            if (targetLower.includes('employable') && !targetLower.includes('less')) {
                encodedLabel = 1; // Employable -> 1
                employableCount++;
            } else if (targetLower.includes('less')) {
                encodedLabel = 0; // LessEmployable -> 0
                lessEmployableCount++;
            } else {
                throw new Error(`Unknown target value: ${targetValue} (row ${index}). Expected 'Employable' or 'LessEmployable'`);
            }
            
            labelsArray.push(encodedLabel);
            
        } catch (error) {
            console.error(`Error processing row ${index}:`, error.message);
            throw error;
        }
    });

    console.log('Enhanced preprocessing completed. Features:', featuresArray.length, 'Labels:', labelsArray.length);
    console.log('Feature dimension:', featuresArray[0].length);
    console.log('Class distribution - Employable:', employableCount, 'LessEmployable:', lessEmployableCount);

    return {
        features: tf.tensor2d(featuresArray),
        labels: tf.tensor1d(labelsArray),
        classDistribution: {
            employable: employableCount,
            lessEmployable: lessEmployableCount
        }
    };
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
        <h4>Performance Metrics (Improved Model)</h4>
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
        <p><strong>Model:</strong> Improved (32-16 neurons with enhanced features)</p>
    `;
}
