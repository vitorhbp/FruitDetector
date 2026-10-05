using TMPro;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.UI;

public class FruitRecognizerWebcam : MonoBehaviour
{
    [Header("Modelo (mobilenetv2-12.onnx)")]
    public ModelAsset modelAsset;
    [Range(0f, 1f)] public float minConfidence = 0.25f;
    public float interval = 0.5f;            // segundos para detecções

    [Header("UI")]
    public RawImage webcamView;
    public TMP_Text infoText;

    const int Size = 224;
    static readonly float[] Mean = { 0.485f, 0.456f, 0.406f };
    static readonly float[] Std  = { 0.229f, 0.224f, 0.225f };

    WebCamTexture webcam;
    Worker worker;
    RenderTexture cropRT;
    Texture2D frameTex;
    float timer;

    void Start()
    {
        if (WebCamTexture.devices.Length == 0) { infoText.text = "Nenhuma webcam encontrada"; enabled = false; return; }

        webcam = new WebCamTexture(WebCamTexture.devices[0].name, 1280, 720);
        webcam.Play();
        webcamView.texture = webcam;
        webcamView.uvRect = new Rect(0f, 0f, 1f, 1f);

        worker = new Worker(ModelLoader.Load(modelAsset), BackendType.CPU);
        cropRT = new RenderTexture(Size, Size, 0, RenderTextureFormat.ARGB32);
        frameTex = new Texture2D(Size, Size, TextureFormat.RGBA32, false);
        infoText.text = "Mostre uma fruta no centro da câmera";
    }

    void OnEnable()
    {
        if (webcam == null) return;
        webcam.Play();
        webcamView.texture = webcam;
        webcamView.uvRect = new Rect(0f, 0f, 1f, 1f);
        infoText.text = "MobileNetV2 ativo. Mostre uma fruta!";
    }

    void OnDisable()
    {
        if (webcam != null && webcam.isPlaying) webcam.Stop();
    }

    void OnDestroy()
    {
        worker?.Dispose();
        if (webcam) webcam.Stop();
        if (cropRT) cropRT.Release();
    }

    void Update()
    {
        if (webcam == null || webcam.width < 100) return;

        timer += Time.deltaTime;
        if (timer < interval) return;
        timer = 0f;

        CaptureCenterCrop();
        var (classId, conf) = Classify();

        if (classId >= 0 && conf >= minConfidence && FruitDatabase.TryGet(classId, out var f))
            infoText.text = $"<b>{f.name}</b> <size=70%>({conf:P0})</size>\n" +
                            $"{f.kcal} kcal / 100 g\n<size=80%>{f.info}</size>";
        else
            infoText.text = $"Procurando fruta... <size=70%>({conf:P0})</size>";
    }

    void CaptureCenterCrop()
    {
        float w = webcam.width, h = webcam.height;
        Vector2 scale = w > h ? new Vector2(h / w, 1f) : new Vector2(1f, w / h);
        Vector2 offset = new((1f - scale.x) / 2f, (1f - scale.y) / 2f);

        Graphics.Blit(webcam, cropRT, scale, offset);
        var prev = RenderTexture.active;
        RenderTexture.active = cropRT;
        frameTex.ReadPixels(new Rect(0, 0, Size, Size), 0, 0);
        frameTex.Apply();
        RenderTexture.active = prev;
    }

    (int classId, float prob) Classify()
    {
        Color32[] px = frameTex.GetPixels32();
        int plane = Size * Size;
        var data = new float[3 * plane];
        for (int i = 0; i < plane; i++)
        {
            data[i]             = (px[i].r / 255f - Mean[0]) / Std[0];
            data[plane + i]     = (px[i].g / 255f - Mean[1]) / Std[1];
            data[2 * plane + i] = (px[i].b / 255f - Mean[2]) / Std[2];
        }

        using var input = new Tensor<float>(new TensorShape(1, 3, Size, Size), data);
        worker.Schedule(input);
        float[] logits = (worker.PeekOutput() as Tensor<float>).DownloadToArray();

        float max = float.MinValue;
        foreach (var l in logits) if (l > max) max = l;
        float sum = 0f;
        foreach (var l in logits) sum += Mathf.Exp(l - max);

        int best = -1; float bestP = 0f;
        foreach (int id in FruitDatabase.ClassIds)
        {
            float p = Mathf.Exp(logits[id] - max) / sum;
            if (p > bestP) { bestP = p; best = id; }
        }
        return (best, bestP);
    }
}