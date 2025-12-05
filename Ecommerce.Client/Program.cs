using Microsoft.ML;
using System;
using System.IO;
using EcommerceML; 

class Program
{
    static void Main(string[] args)
    {
        var mlContext = new MLContext();
        string modelPath = "EcommerceModel.zip";

        // Перевірка наявності файлу моделі
        if (!File.Exists(modelPath))
        {
            Console.WriteLine($"Error: File '{modelPath}' not found. Copy it from the Trainer project output.");
            return;
        }

        // 1. Завантаження моделі
        DataViewSchema modelSchema;
        ITransformer loadedModel = mlContext.Model.Load(modelPath, out modelSchema);

        // 2. Створення двигуна для одиночних прогнозів
        var predictionEngine = mlContext.Model.CreatePredictionEngine<EcommerceData, EcommercePrediction>(loadedModel);

        Console.WriteLine("--- Ecommerce Purchase Predictor (Kaggle Dataset) ---");

        // --- ТЕСТОВИЙ ПРИКЛАД ---
        // Спробуємо спрогнозувати поведінку користувача, який:
        // - Новий відвідувач (New_Visitor)
        // - Переглянув 10 товарів
        // - PageValues (метрика зацікавленості) висока = 50.0
        var sampleData = new EcommerceData
        {
            VisitorType = "New_Visitor",
            ProductRelated = 10,
            PageValues = 50.0f
        };

        // 3. Прогноз
        var prediction = predictionEngine.Predict(sampleData);

        Console.WriteLine($"\nInput: Type={sampleData.VisitorType}, Viewed={sampleData.ProductRelated}, PageValue={sampleData.PageValues}");
        Console.WriteLine($"Will buy? (Prediction): {prediction.Prediction}"); // True або False
        Console.WriteLine($"Confidence (Probability): {prediction.Probability:P2}");

        // Додатковий приклад (негативний)
        var negativeSample = new EcommerceData { VisitorType = "Returning_Visitor", ProductRelated = 2, PageValues = 0.0f };
        var negPrediction = predictionEngine.Predict(negativeSample);

        Console.WriteLine($"\nInput: Type={negativeSample.VisitorType}, Viewed={negativeSample.ProductRelated}, PageValue={negativeSample.PageValues}");
        Console.WriteLine($"Will buy? (Prediction): {negPrediction.Prediction}");
        Console.WriteLine($"Confidence (Probability): {negPrediction.Probability:P2}");

        Console.ReadLine();
    }
}