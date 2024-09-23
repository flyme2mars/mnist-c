#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Constants
#define MAX_LINE_LENGTH 10000
#define TRAINING_SAMPLES 42000
#define TEST_SAMPLES 28000
#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define EPOCHS 5
#define LEARNING_RATE 0.05

typedef struct{
   int inputSize;
   int hiddenSize;
   int outputSize;
   double **hiddenWeights;
   double **outputWeights;
   double *hiddenBias;
   double *outputBias;
} NeuralNetwork;

// RELU implimentation
double relu(double x){
   return (x > 0) ? x : 0;
}

// RELU prime implementation
double reluDerivative(double x){
   return (x > 0) ? 1 : 0;
}

// Softmax implementation
void softmax(double *input, int size){
   // finding the max of input array
   double max = input[0];
   for (int i = 1; i < size; i++){
      if (input[i] > max)
         max = input[i];
   }

   // find the sum of exponential of input array after normalizing
   double sum = 0.0;
   for (int i = 0; i < size; i++){
      input[i] = exp(input[i] - max);
      sum += input[i];
   }

   // dividing each element by the sum of exponentials
   for (int i = 0; i < size; i++){
      input[i] /= sum;
   }
}

NeuralNetwork *createNeuralNetwork(int inputSize, int hiddenSize, int outputSize){
   // Allocating memory for NeuralNetwork
   NeuralNetwork *nn = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
   if (nn == NULL) return NULL;

   // Initializing input, hidden and output sizes..
   nn->inputSize = inputSize;
   nn->hiddenSize = hiddenSize;
   nn->outputSize = outputSize;

   // Allocating memory for weights(row) in hidden layer
   nn->hiddenWeights = (double **)malloc(hiddenSize * sizeof(double *));

   // If error allocating memory, free up memory and return NULL
   if (nn->hiddenWeights == NULL){
      free(nn);
      return NULL;
   }

   for (int i = 0; i < hiddenSize; i++){
      // Allocate memory for weights(column) in hidden layer
      nn->hiddenWeights[i] = (double *)malloc(inputSize * sizeof(double));

      // If error allocating memory, free up memory and return NULL
      if (nn->hiddenWeights[i] == NULL){
         for (int j = 0; j < i; j++) free(nn->hiddenWeights[j]);
         free(nn->hiddenWeights);
         free(nn);
         return NULL;
      }

      // Initialize hidden weights with random numbebers between -1 and 1
      for (int j = 0; j < inputSize; j++){
         nn->hiddenWeights[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
      }
   }

   // Allocating memory for weights(row) in output layer
   nn->outputWeights = (double **)malloc(outputSize * sizeof(double *));

   // If error allocating memory, free up memory and return NULL
   if (nn->outputWeights == NULL){
      for (int j = 0; j < hiddenSize; j++) free(nn->hiddenWeights[j]);
      free(nn->hiddenWeights);
      free(nn);
      return NULL;
   }

   for (int i = 0; i < outputSize; i++){
      // Allocating memory for weights(column) in output layer
      nn->outputWeights[i] = (double *)malloc(hiddenSize * sizeof(double));

      // If error allocating memory, free up memory and return NULL
      if (nn->outputWeights[i] == NULL){
         for (int j = 0; j < i; j++) free(nn->outputWeights[j]);
         for (int j = 0; j < hiddenSize; j++) free(nn->hiddenWeights[j]);
         free(nn->outputWeights);
         free(nn->hiddenWeights);
         free(nn);
         return NULL;
      }

      // Initialize output weights with random numbebers between -1 and 1
      for (int j = 0; j < hiddenSize; j++){
         nn->outputWeights[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
      }
   }

   // Allocating memory for hidden and output biases
   nn->hiddenBias = (double *)malloc(hiddenSize * sizeof(double));
   nn->outputBias = (double *)malloc(outputSize * sizeof(double));

   // If error allocating memory, free up memory and return NULL
   if (nn->hiddenBias == NULL || nn->outputBias == NULL){
      for (int i = 0; i < outputSize; i++) free(nn->outputWeights[i]);
      for (int i = 0; i < hiddenSize; i++) free(nn->hiddenWeights[i]);
      free(nn->outputWeights);
      free(nn->hiddenWeights);
      free(nn->outputBias);
      free(nn->hiddenBias);
      free(nn);
      return NULL;
   }

   // Initialize biases with random numbers between -1 and 1
   for (int i = 0; i < hiddenSize; i++){
      nn->hiddenBias[i] = ((double)rand() / RAND_MAX) * 2 - 1;
   }
   for (int i = 0; i < outputSize; i++){
      nn->outputBias[i] = ((double)rand() / RAND_MAX) * 2 - 1;
   }

   // Return nn
   return nn;
}

void forwardPropagation(NeuralNetwork *nn, double *input, double *hiddenActivation, double *outputActivation){
   // Calculating hidden layer activation
   for (int i = 0; i < nn->hiddenSize; i++){
      hiddenActivation[i] = 0;
      for (int j = 0; j < nn->inputSize; j++){
         hiddenActivation[i] += input[j] * nn->hiddenWeights[i][j];
      }
      hiddenActivation[i] = relu(hiddenActivation[i] + nn->hiddenBias[i]);
   }

   // Calculating output layer activation
   for (int i = 0; i < nn->outputSize; i++){
      outputActivation[i] = 0;
      for (int j = 0; j < nn->hiddenSize; j++){
         outputActivation[i] += hiddenActivation[j] * nn->outputWeights[i][j];
      }
      outputActivation[i] += nn->outputBias[i];
   }

   // Softmaxing the output
   softmax(outputActivation, nn->outputSize);
}

void backward(NeuralNetwork *nn, double *input, double *hiddenActivation, double *outputActivation, int label, double lr){
   // Calculating output error...
   double *outputError = (double *)malloc(nn->outputSize * sizeof(double));
   if (outputError == NULL) return;
   for (int i = 0; i < nn->outputSize; i++){
      outputError[i] = (i == label) ? outputActivation[i] - 1: outputActivation[i];
   }

   // Calculating hidden error...
   double *hiddenError = (double *)malloc(nn->hiddenSize * sizeof(double));
   if (hiddenError == NULL) return;
   for (int i = 0; i < nn->hiddenSize; i++){
      for (int j = 0; j < nn->outputSize; j++){
         hiddenError[i] += nn->outputWeights[j][i] * outputError[j];
      }
      hiddenError[i] *= reluDerivative(hiddenActivation[i]);
   }

   // Updating output weights...
   for (int i = 0; i < nn->outputSize; i++){
      for (int j = 0; j < nn->hiddenSize; j++){
         nn->outputWeights[i][j] -= lr * hiddenActivation[j] * outputError[i];
      }
   }

   // Updating hidden weights...
   for (int i = 0; i < nn->hiddenSize; i++){
      for (int j = 0; j < nn->inputSize; j++){
         nn->hiddenWeights[i][j] -= lr * input[j] * hiddenError[i];
      }
   }

   // Updating biases...
   for (int i = 0; i < nn->outputSize; i++) nn->outputBias[i] -= lr * outputError[i];
   for (int i = 0; i < nn->hiddenSize; i++) nn->hiddenBias[i] -= lr * hiddenError[i];

   // Freeing memory....
   free(outputError);
   free(hiddenError);
}

void trainNetwork(NeuralNetwork *nn, double **trainingData, int *labels, int numSamples, int epochs, double lr){
   // Allocating memory for hidden and output activations
   double *hiddenActivation = (double *)malloc(nn->hiddenSize * sizeof(double));
   if (hiddenActivation == NULL) return;
   double *outputActivation = (double *)malloc(nn->outputSize * sizeof(double));
   if (outputActivation == NULL) return;

   // Training...
   for (int epoch = 0; epoch < epochs; epoch++){
      double totalLoss = 0.0;
      for (int i = 0; i < numSamples; i++ ){
         if (trainingData[i] == NULL) return;

         forwardPropagation(nn, trainingData[i], hiddenActivation, outputActivation);

         if (labels[i] < 0 || labels[i] >= nn->outputSize) return;

         // Calculating cross entropy loss
         totalLoss -= log(outputActivation[labels[i]]);

         backward(nn, trainingData[i], hiddenActivation, outputActivation, labels[i], lr);
      }
      printf("Epoch %d/%d completed, Average Loss: %f\n", epoch + 1, epochs, totalLoss / numSamples);
   }

   // Freeing memory...
   free(hiddenActivation);
   free(outputActivation);
}

void freeNeuralNetwork(NeuralNetwork *nn){
   if (nn == NULL) return;

   // Freeing hidden weights
   for (int i = 0; i < nn->hiddenSize; i ++) free(nn->hiddenWeights[i]);
   free(nn->hiddenWeights);

   // Freeing output weights
   for (int i = 0; i < nn->outputSize; i++) free(nn->outputWeights[i]);
   free(nn->outputWeights);

   // Freeing biases
   free(nn->hiddenBias);
   free(nn->outputBias);

   // Freeing neural network...
   free(nn);
}

double calculateAccuracy(NeuralNetwork *nn, double **images, int *labels, int numSamples){
      // Allocating memory for hidden and output activations
   double *hiddenActivation = (double *)malloc(nn->hiddenSize * sizeof(double));
   if (hiddenActivation == NULL) return 0.0;
   double *outputActivation = (double *)malloc(nn->outputSize * sizeof(double));
   if (outputActivation == NULL) return 0.0;

   int correct = 0;

   for (int i = 0; i < numSamples; i++){
      forwardPropagation(nn, images[i], hiddenActivation, outputActivation);
      int predictedLabel = 0;
      double maxOutput = outputActivation[0];
      for (int j = 1; j < nn->outputSize; j ++){
         if (outputActivation[j] > maxOutput){
            maxOutput = outputActivation[j];
            predictedLabel = j;
         }
      }
      if (predictedLabel == labels[i]) correct ++;
   }

      // Freeing memory...
   free(hiddenActivation);
   free(outputActivation);

   return (double)correct / numSamples;
}


int main(){
   // Setting random seed
   srand(time(NULL));

   // Initialization...
   FILE *file;
   char line[MAX_LINE_LENGTH];
   char *token;

   int trainingSize = TRAINING_SAMPLES;
   int testSize = TEST_SAMPLES;

   // Allocating memory for training/test images and labels
   double **trainingImages = (double **)malloc(TRAINING_SAMPLES * sizeof(double *));
   int *trainingLabels = (int *)malloc(TRAINING_SAMPLES * sizeof(int));
   double **testImages = (double **)malloc(TEST_SAMPLES * sizeof(double *));
   int *testLabels = (int *)malloc(TEST_SAMPLES * sizeof(int));

   for (int i = 0; i < TRAINING_SAMPLES; i++) trainingImages[i] = (double *)malloc(INPUT_SIZE * sizeof(double));
   
   for (int i = 0; i < TEST_SAMPLES; i++) testImages[i] = (double *)malloc(INPUT_SIZE * sizeof(double));

   // Loading training data
   file = fopen("dataset/train.csv", "r");
   if (file == NULL){
      printf("Error opening training file\n");
      exit(1);
   }
 
   fgets(line, MAX_LINE_LENGTH, file);

   for (int i = 0; i < trainingSize; i++){
      fgets(line, MAX_LINE_LENGTH, file);
      token = strtok(line, ",");
      trainingLabels[i] = atoi(token);

      for (int j = 0; j < INPUT_SIZE; j++){
         token = strtok(NULL, ",");
         trainingImages[i][j] = atof(token) / 255.0;
      }
   }

   fclose(file);

   // Loading test data
   file = fopen("dataset/test.csv", "r");
   if (file == NULL){
      printf("Error opening test file\n");
      exit(1);
   }

   fgets(line, MAX_LINE_LENGTH, file);

   for (int i = 0; i < testSize; i++){
      fgets(line, MAX_LINE_LENGTH, file);

      // testLabels[i] = -1;

      for (int j = 0; j < 28 * 28; j++){
         token = strtok(j == 0 ? line : NULL, ",");
         testImages[i][j] = atoi(token) / 255.0;
      }
   }

   fclose(file);

   NeuralNetwork *nn = createNeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
   trainNetwork(nn, trainingImages, trainingLabels, TRAINING_SAMPLES, EPOCHS, LEARNING_RATE);

   double trainingAccuracy = calculateAccuracy(nn, trainingImages, trainingLabels, TRAINING_SAMPLES);
   printf("Training accuracy: %.2f%%\n", trainingAccuracy * 100);

   // double testAccuracy = calculateAccuracy(nn, testImages, testLabels, TEST_SAMPLES);
   // printf("Test accuracy: %.2f%%\n", testAccuracy * 100);

   // Visualizing an Image...
   // int x;
   // x = 5;
   // printf("Image: \n");
   // for (int i = 0; i < INPUT_SIZE; i++){
   //    if (trainingImages[x][i] > 0.5) printf("██");
   //    else printf("  ");
   //    if (i % 28 == 27) printf("\n");
   // }

   // printf("Label: %d\n", trainingLabels[x]);


   // Freeing......
   freeNeuralNetwork(nn);
   for (int i = 0; i < TRAINING_SAMPLES; i++){
      free(trainingImages[i]);
   }
   for (int i = 0; i < TEST_SAMPLES; i++){
      free(testImages[i]);
   }
   free(trainingImages);
   free(trainingLabels);
   free(testImages);
   free(testLabels);
}
