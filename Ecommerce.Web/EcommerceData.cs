using Microsoft.ML.Data;

namespace EcommerceML
{
    public class EcommerceData
    {
        // Колонка 4: ProductRelated - Кількість переглянутих сторінок товарів
        [LoadColumn(4)]
        public float ProductRelated { get; set; }

        // Колонка 8: PageValues - Середня цінність відвіданих сторінок
        [LoadColumn(8)]
        public float PageValues { get; set; }

        // Колонка 15: VisitorType - "Returning_Visitor" або "New_Visitor"
        [LoadColumn(15)]
        public string? VisitorType { get; set; }

        // Колонка 17: Revenue - Цільова змінна (TRUE/FALSE)
        [LoadColumn(17), ColumnName("Label")]
        public bool Revenue { get; set; }
    }

    public class EcommercePrediction
    {
        // ONNX повертає масив, тому змінюємо bool на bool[]
        [ColumnName("PredictedLabel")]
        public bool[] Prediction { get; set; }

        // Score в ONNX теж часто є масивом ймовірностей, тому float[]
        [ColumnName("Score")]
        public float[] Score { get; set; }

        // Probability теж може бути масивом
        [ColumnName("Probability")]
        public float[] Probability { get; set; }
    }
}
