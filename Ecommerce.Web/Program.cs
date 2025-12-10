using Microsoft.ML;
using Microsoft.Extensions.ML;
using EcommerceML;
// Додаємо простори імен для Swagger
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

var builder = WebApplication.CreateBuilder(args);

// --- 1. РЕЄСТРАЦІЯ СТАНДАРТНИХ СЕРВІСІВ ---
builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

// --- 2. ЗАВАНТАЖЕННЯ ONNX МОДЕЛІ ВРУЧНУ ---
// Ми робимо це тут, щоб уникнути помилки "Failed to open zip archive"
// Створюємо глобальний MLContext
var mlContext = new MLContext();

string modelPath = "EcommerceModel.onnx";

// Створюємо пайплайн: Завантаження ONNX -> Перетворення
var onnxPipeline = mlContext.Transforms.ApplyOnnxModel(modelPath);

// "Тренуємо" на порожніх даних, щоб отримати готову модель (ITransformer)
var emptyData = mlContext.Data.LoadFromEnumerable(new List<EcommerceData>());
var onnxModel = onnxPipeline.Fit(emptyData);

// --- 3. РЕЄСТРАЦІЯ МОДЕЛІ ЯК СИНГЛТОН ---
// Замість PredictionEnginePool (який вередує з ONNX), ми реєструємо саму модель
builder.Services.AddSingleton<ITransformer>(onnxModel);
builder.Services.AddSingleton<MLContext>(mlContext);

var app = builder.Build();

// --- 4. НАЛАШТУВАННЯ HTTP ---
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseAuthorization();
app.MapControllers();

app.Run();