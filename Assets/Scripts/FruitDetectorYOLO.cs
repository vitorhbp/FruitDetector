using System.Collections.Generic;
using TMPro;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.UI;

public class FruitDetectorYOLO : MonoBehaviour
{
    [Header("Modelo (yolo11n.onnx)")]
    public ModelAsset modelAsset;
    public BackendType backend = BackendType.GPUCompute;
    [Range(0f, 1f)] public float scoreThreshold = 0.3f;
    [Range(0f, 1f)] public float iouThreshold = 0.5f;
    public float interval = 0.15f;                       // tempo para detecções

    [Header("UI")]
    public RawImage webcamView;
    public TMP_Text infoText;
    public Color boxColor = new(0.2f, 1f, 0.3f);
    public float lineWidth = 4f;

    const int Size = 640;
    // IDs do COCO: 46 = banana, 47 = maçã, 48 = sanduíche, 49 = laranja
    const int FirstClass = 46, ClassCount = 4;

    WebCamTexture webcam;
    Worker worker;
    RenderTexture inputRT;
    Tensor<float> centersToCorners;
    float timer;

    class Box { public RectTransform root; public TMP_Text label; }
    readonly List<Box> boxes = new();

    void Start()
    {
        if (WebCamTexture.devices.Length == 0) { infoText.text = "Nenhuma webcam encontrada"; enabled = false; return; }

        webcam = new WebCamTexture(WebCamTexture.devices[0].name, 1280, 720);
        webcam.wrapMode = TextureWrapMode.Clamp;
        webcam.Play();
        webcamView.texture = webcam;

        var fitter = webcamView.GetComponent<AspectRatioFitter>();
        if (fitter) Destroy(fitter);

        inputRT = new RenderTexture(Size, Size, 0, RenderTextureFormat.ARGB32);
        BuildWorker();
        infoText.text = "Mostre uma banana, maçã ou laranja";
    }

    void BuildWorker()
    {
        var model = ModelLoader.Load(modelAsset);

        centersToCorners = new Tensor<float>(new TensorShape(4, 4), new float[]
        {
            1, 0, 1, 0,
            0, 1, 0, 1,
            -0.5f, 0, 0.5f, 0,
            0, -0.5f, 0, 0.5f
        });

        var graph = new FunctionalGraph();
        var inputs = graph.AddInputs(model);
        var output = Functional.Forward(model, inputs)[0];

        var boxCoords = output[0, 0..4, ..].Transpose(0, 1);
        var fruitScores = output[0, (4 + FirstClass)..(4 + FirstClass + ClassCount), ..];
        var scores = Functional.ReduceMax(fruitScores, 0);
        var classIds = Functional.ArgMax(fruitScores, 0);
        var corners = Functional.MatMul(boxCoords, Functional.Constant(centersToCorners));
        var keep = Functional.NMS(corners, scores, iouThreshold, scoreThreshold);

        worker = new Worker(graph.Compile(
            Functional.IndexSelect(boxCoords, 0, keep),
            Functional.IndexSelect(classIds, 0, keep),
            Functional.IndexSelect(scores, 0, keep)), backend);
    }

    void OnDestroy()
    {
        worker?.Dispose();
        centersToCorners?.Dispose();
        if (webcam) webcam.Stop();
        if (inputRT) inputRT.Release();
    }

    void Update()
    {
        if (webcam == null || webcam.width < 100) return;

        Rect area = webcamView.rectTransform.rect;
        float camAspect = (float)webcam.width / webcam.height;
        float viewAspect = area.width / Mathf.Max(1f, area.height);
        Rect uv = viewAspect > camAspect
            ? new Rect(0f, (1f - camAspect / viewAspect) / 2f, 1f, camAspect / viewAspect)
            : new Rect((1f - viewAspect / camAspect) / 2f, 0f, viewAspect / camAspect, 1f);
        webcamView.uvRect = uv;

        timer += Time.deltaTime;
        if (timer < interval) return;
        timer = 0f;

        if (viewAspect >= 1f) { contentW = Size; contentH = Size / viewAspect; }
        else                  { contentH = Size; contentW = Size * viewAspect; }
        padX = (Size - contentW) / 2f;
        padY = (Size - contentH) / 2f;

        Vector2 scale = new(uv.width * Size / contentW, uv.height * Size / contentH);
        Vector2 offset = new(uv.x - padX / Size * scale.x, uv.y - padY / Size * scale.y);
        Graphics.Blit(webcam, inputRT, scale, offset);

        Detect();
    }

    float contentW = Size, contentH = Size, padX, padY;

    void Detect()
    {
        using var input = new Tensor<float>(new TensorShape(1, 3, Size, Size));
        TextureConverter.ToTensor(inputRT, input, default);
        worker.Schedule(input);

        using var coords = (worker.PeekOutput("output_0") as Tensor<float>).ReadbackAndClone();
        using var ids    = (worker.PeekOutput("output_1") as Tensor<int>).ReadbackAndClone();
        using var scores = (worker.PeekOutput("output_2") as Tensor<float>).ReadbackAndClone();

        int n = coords.shape[0];
        Rect area = webcamView.rectTransform.rect;
        float sx = area.width / contentW, sy = area.height / contentH;

        var lines = new List<string>();
        var shown = new HashSet<int>();
        int used = 0;

        for (int i = 0; i < n; i++)
        {
            int cocoId = ids[i] + FirstClass;
            if (!FruitDatabase.TryGet(cocoId, out var fruit)) continue;
            float score = scores[i];

            var box = GetBox(used++);
            box.root.anchoredPosition = new Vector2((coords[i, 0] - padX) * sx, -(coords[i, 1] - padY) * sy);
            box.root.sizeDelta = new Vector2(coords[i, 2] * sx, coords[i, 3] * sy);
            box.label.text = $"<mark=#000000AA> {fruit.name} {score:P0} </mark>";
            box.label.rectTransform.localScale = new Vector3(Mathf.Sign(webcamView.rectTransform.localScale.x), 1, 1);

            if (shown.Add(cocoId))
                lines.Add($"<b>{fruit.name}</b> <size=70%>({score:P0})</size> - {fruit.kcal} kcal / 100 g\n<size=80%>{fruit.info}</size>");
        }

        for (int i = used; i < boxes.Count; i++) boxes[i].root.gameObject.SetActive(false);

        infoText.text = lines.Count > 0 ? string.Join("\n", lines) : "Procurando fruta...";
    }

    // Caixas
    Box GetBox(int index)
    {
        while (boxes.Count <= index) boxes.Add(CreateBox());
        var b = boxes[index];
        b.root.gameObject.SetActive(true);
        return b;
    }

    Box CreateBox()
    {
        var go = new GameObject("FruitBox", typeof(RectTransform));
        var rt = (RectTransform)go.transform;
        rt.SetParent(webcamView.rectTransform, false);
        rt.anchorMin = rt.anchorMax = new Vector2(0, 1);
        rt.pivot = new Vector2(0.5f, 0.5f);

        AddLine(rt, new Vector2(0, 1), new Vector2(1, 1), new Vector2(0.5f, 1), new Vector2(0, lineWidth)); // topo
        AddLine(rt, new Vector2(0, 0), new Vector2(1, 0), new Vector2(0.5f, 0), new Vector2(0, lineWidth)); // base
        AddLine(rt, new Vector2(0, 0), new Vector2(0, 1), new Vector2(0, 0.5f), new Vector2(lineWidth, 0)); // esquerda
        AddLine(rt, new Vector2(1, 0), new Vector2(1, 1), new Vector2(1, 0.5f), new Vector2(lineWidth, 0)); // direita

        var labelGo = new GameObject("Label", typeof(RectTransform));
        var lrt = (RectTransform)labelGo.transform;
        lrt.SetParent(rt, false);
        lrt.anchorMin = lrt.anchorMax = new Vector2(0, 1);
        lrt.pivot = new Vector2(0, 0);
        lrt.anchoredPosition = new Vector2(0, 4);
        lrt.sizeDelta = new Vector2(500, 40);

        var label = labelGo.AddComponent<TextMeshProUGUI>();
        label.fontSize = 28;
        label.color = boxColor;
        label.raycastTarget = false;
        label.textWrappingMode = TextWrappingModes.NoWrap;

        return new Box { root = rt, label = label };
    }

    void AddLine(RectTransform parent, Vector2 aMin, Vector2 aMax, Vector2 pivot, Vector2 size)
    {
        var go = new GameObject("Line", typeof(RectTransform));
        var rt = (RectTransform)go.transform;
        rt.SetParent(parent, false);
        rt.anchorMin = aMin; rt.anchorMax = aMax; rt.pivot = pivot;
        rt.anchoredPosition = Vector2.zero;
        rt.sizeDelta = size;
        var img = go.AddComponent<Image>();
        img.color = boxColor;
        img.raycastTarget = false;
    }
}