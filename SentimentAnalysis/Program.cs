using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Trainers;df
using Microsoft.ML.Transforms.Text;

namespace SentimentAnalysis
{
    class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled.txt");
        static void Main(string[] args)
        {
            var mlContext = new MLContext();
            var splitDataView = LoadData(mlContext);
            var model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);
            Evaluate(mlContext, model, splitDataView.TestSet);
            UseModelWithSingleItem(mlContext, model);
            UseModelWithBatchItems(mlContext, model);
        }
        public static ITransformer BuildAndTrainModel(MLContext mLContext,IDataView splitTrainSet)
        {
            var estimator = mLContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
                .Append(mLContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));


            Console.WriteLine("=============== Create and Train the Model ===============");
            var model = estimator.Fit(splitTrainSet);
            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();

            return model;
        }
        public static void Evaluate(MLContext mLContext,ITransformer mode,IDataView splitTestSet)
        {
            Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
            var predictions = mode.Transform(splitTestSet);
            var metrics = mLContext.BinaryClassification.Evaluate(predictions, "Label");

            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");
        }
        private static void UseModelWithSingleItem(MLContext mlContext,ITransformer model)
        {
            var predictionFunction = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
            var sampleStatement = new SentimentData
            {
                SentimentText = "This was a very bad steak"
            };
            var resultPrediction = predictionFunction.Predict(sampleStatement);

            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

            Console.WriteLine();
            Console.WriteLine($"Sentiment: {resultPrediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability} ");

            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();
        }
        public static void UseModelWithBatchItems(MLContext mLContext,ITransformer model)
        {
            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                {
                    SentimentText ="This was a horrible meal"
                },
                new SentimentData
                {
                    SentimentText="I love this spaghetti."
                }
            };

            var batchComments = mLContext.Data.LoadFromEnumerable(sentiments);
            var predictions = model.Transform(batchComments);
            var predictedResults = mLContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);
            Console.WriteLine();

            Console.WriteLine("=============== Prediction Test of loaded model with multiple samples ===============");
            
            foreach(SentimentPrediction prediction in predictedResults)
            {
                Console.WriteLine($"Sentiment: {prediction.SentimentText} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability} ");
            }
            Console.WriteLine("=============== End of predictions ===============");
        }
        public static TrainTestData LoadData(MLContext mlContext)
        {
            var dataview = mlContext.Data.LoadFromTextFile<SentimentData>(_dataPath, hasHeader:false);
            var splitDataView = mlContext.Data.TrainTestSplit(dataview, testFraction:0.2);
            return splitDataView;
        }
    }
}
