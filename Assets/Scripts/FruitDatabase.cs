using System.Collections.Generic;

public struct FruitInfo
{
    public string name;
    public int kcal;
    public string info;
    public FruitInfo(string n, int k, string i) { name = n; kcal = k; info = i; }
}

// Informações das frutas, indexadas pelo ID da classe de cada modelo
// IDs 46-49 são para YOLO (COCO)
// IDs 948-957 são para MobileNetV2
public static class FruitDatabase
{
    static readonly Dictionary<int, FruitInfo> fruits = new()
    {
        // YOLOn11
        { 46,  new FruitInfo("Banana",          89, "Rica em potássio; energia rápida") },
        { 47,  new FruitInfo("Maçã",            52, "Rica em fibras (pectina); ajuda na saciedade") },
        { 49,  new FruitInfo("Laranja",         47, "Fonte de vitamina C e folato") },

        // MobileNetV2
        { 948, new FruitInfo("Maçã",            52, "Rica em fibras (pectina); ajuda na saciedade") },
        { 949, new FruitInfo("Morango",         32, "Muita vitamina C; baixa caloria") },
        { 950, new FruitInfo("Laranja",         47, "Fonte de vitamina C e folato") },
        { 951, new FruitInfo("Limão",           29, "Vitamina C; ácido cítrico") },
        { 952, new FruitInfo("Figo",            74, "Fibras, potássio e cálcio") },
        { 953, new FruitInfo("Abacaxi",         50, "Contém bromelina (enzima digestiva)") },
        { 954, new FruitInfo("Banana",          89, "Rica em potássio; energia rápida") },
        { 955, new FruitInfo("Jaca",            95, "Fibras, vitamina C e potássio") },
        { 956, new FruitInfo("Fruta-do-conde",  94, "Vitamina C e fibras; bem doce") },
        { 957, new FruitInfo("Romã",            83, "Rica em antioxidantes (polifenóis)") },
    };

    public static IEnumerable<int> ClassIds => fruits.Keys;

    public static bool TryGet(int classId, out FruitInfo info) => fruits.TryGetValue(classId, out info);
}