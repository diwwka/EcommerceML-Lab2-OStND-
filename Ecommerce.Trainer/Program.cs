using EcommerceML; 
using Microsoft.ML;
using System;

class Program
{
    static void Main(string[] args)
    {
        var mlContext = new MLContext(seed: 0);

        // Щлях до завантаженого файлу з Kaggle
        var dataPath = "online_shoppers_intention.csv";

        Console.WriteLine("Loading data...");
        // Важливо: hasHeader: true, бо в CSV є перший рядок з назвами
        IDataView dataView = mlContext.Data.LoadFromTextFile<EcommerceData>(dataPath, hasHeader: true, separatorChar: ',');

        // Розбиваємо: 80% на навчання, 20% на тест
        var split = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

        // --- БУДУЄМО ПАЙПЛАЙН ---
        var pipeline = mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VisitorTypeEncoded", inputColumnName: nameof(EcommerceData.VisitorType))
            .Append(mlContext.Transforms.Concatenate("Features", "VisitorTypeEncoded", "PageValues", "ProductRelated"))
            // Використовуємо FastTreeBinaryTrainer (як у методичці для бінарної класифікації)
            .Append(mlContext.BinaryClassification.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features"));

        // --- ТРЕНУВАННЯ ---
        Console.WriteLine("Start Training...");
        var model = pipeline.Fit(split.TrainSet);

        // --- ОЦІНКА ---
        Console.WriteLine("Evaluating...");
        var predictions = model.Transform(split.TestSet);
        var metrics = mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: "Label");

        Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
        Console.WriteLine($"F1 Score: {metrics.F1Score:P2}"); // F1 важливий для незбалансованих класів (покупок завжди менше)
        Console.WriteLine($"Area Under ROC Curve: {metrics.AreaUnderRocCurve:P2}");

        Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());

        // --- ЗБЕРЕЖЕННЯ ---
        // Зберігаємо модель у ZIP файл
        mlContext.Model.Save(model, dataView.Schema, "EcommerceModel.zip");
        Console.WriteLine("Model Saved as 'EcommerceModel.zip'!");

        // ... (весь попередній код тренування з Лаб 2) ...

        Console.WriteLine("Збереження у форматі ONNX...");

        // Створюємо потік для збереження файлу
        using (var stream = File.Create("EcommerceModel.onnx"))
        {
            // Конвертуємо та зберігаємо модель
            // Важливо: ML.NET потребує знати вхідну схему (dataView)
            mlContext.Model.ConvertToOnnx(model, dataView, stream);
        }

        Console.WriteLine("Model Saved as 'EcommerceModel.onnx'!");
    }
}