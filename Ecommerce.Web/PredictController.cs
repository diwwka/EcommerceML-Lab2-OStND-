using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using EcommerceML;

namespace Ecommerce.Web.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class PredictController : ControllerBase
    {
        private readonly ITransformer _model;
        private readonly MLContext _mlContext;

        // Отримуємо завантажену ONNX модель через конструктор
        public PredictController(ITransformer model, MLContext mlContext)
        {
            _model = model;
            _mlContext = mlContext;
        }

        [HttpPost]
        public ActionResult<EcommercePrediction> Post([FromBody] EcommerceData input)
        {
            if (!ModelState.IsValid)
            {
                return BadRequest();
            }

            // Створюємо двигун передбачення "на льоту" для конкретного запиту
            // Це надійний спосіб для ONNX моделей
            var predictionEngine = _mlContext.Model.CreatePredictionEngine<EcommerceData, EcommercePrediction>(_model);

            // Отримання прогнозу
            var prediction = predictionEngine.Predict(input);

            return Ok(prediction);
        }
    }
}