﻿using System;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Microsoft.ML;
using System.Threading.Tasks;

namespace ConsoleApp
{
    class Program
    {
         // <Snippet2>
        const string DataPath = @"..\..\Data\taxi-fare-train.csv";
        const string TestDataPath = @"..\..\Data\taxi-fare-test.csv";
        const string ModelPath = @"..\..\Models\Model.zip";
        // </Snippet2>
 
        // <Snippet8>
        static void Main(string[] args)
        // </Snippet8>
        {
            Console.WriteLine(Environment.CurrentDirectory);
            // <Snippet7>
            PredictionModel<TaxiTrip, TaxiTripFarePrediction> model = Train();
            // </Snippet7>
 
            // <Snippet10>
            Evaluate(model);
            // </Snippet10>
 
            // <Snippet16>
            var prediction = model.Predict(TestTrips.Trip1);
            Console.WriteLine("Predicted fare: {0}, actual fare: 29.5", prediction.fare_amount);
            // </Snippet16>
        }
 
        // <Snippet6>
        public static PredictionModel<TaxiTrip, TaxiTripFarePrediction> Train()
        // </Snippet6>
        {
            // <Snippet3>
            var pipeline = new LearningPipeline
            {
                new TextLoader<TaxiTrip>(DataPath, useHeader: true, separator: ","),
                new ColumnCopier(("fare_amount", "Label")),
                new CategoricalOneHotVectorizer("vendor_id",
                                            "rate_code",
                                            "payment_type"),
                new ColumnConcatenator("Features",
                                                "vendor_id",
                                                "rate_code",
                                                "passenger_count",
                                                "trip_distance",
                                                "payment_type"),
                new FastTreeRegressor()
            };
            // </Snippet3>
 
            // <Snippet4>
            PredictionModel<TaxiTrip, TaxiTripFarePrediction> model = pipeline.Train<TaxiTrip, TaxiTripFarePrediction>();
            // </Snippet4>
            // <Snippet5>
            model.WriteAsync(ModelPath);
            return model;
            // </Snippet5>
        }
 
        // <Snippet11>
        private static void Evaluate(PredictionModel<TaxiTrip, TaxiTripFarePrediction> model)
        // </Snippet11>
        {
            // <Snippet12>
            var testData = new TextLoader<TaxiTrip>(TestDataPath, useHeader: true, separator: ",");
            // </Snippet12>
 
            // <Snippet13>
            var evaluator = new RegressionEvaluator();
            RegressionMetrics metrics = evaluator.Evaluate(model, testData);
            // </Snippet13>
 
            // <Snippet14>
            // Rms should be around 2.795276
            Console.WriteLine("Rms=",metrics.Rms);
            // </Snippet14>
            // <Snippet15>
            Console.WriteLine("RSquared = ",metrics.RSquared);
            // </Snippet15>
        }
    }
}
