using TMPro;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.UI;

// Alternar entre os dois modelos (MobileNetV2 e YOLO11n) durante a execução
// Botão no canto superior direito da tela
// Tab: troca o modelo
// Esc: fecha o aplicativo
public class ModelSwitcher : MonoBehaviour
{
    [Header("Componentes dos modelos")]
    public FruitRecognizerWebcam mobileNet;
    public FruitDetectorYOLO yolo;

    [Header("Opções")]
    public bool startWithYolo = true;
    public Key switchKey = Key.Tab;
    public TMP_Text infoText;

    bool usingYolo;
    TMP_Text buttonLabel;

    void Awake()
    {
        if (mobileNet) mobileNet.enabled = false;
        if (yolo) yolo.enabled = false;
    }

    void Start()
    {
        CreateButton();
        Apply(startWithYolo);
    }

    void Update()
    {
        var kb = Keyboard.current;
        if (kb == null) return;
        if (kb[switchKey].wasPressedThisFrame) Toggle();
        if (kb[Key.Escape].wasPressedThisFrame) Application.Quit();
    }

    public void Toggle() => Apply(!usingYolo);

    void Apply(bool yoloOn)
    {
        usingYolo = yoloOn;

        mobileNet.enabled = false;
        yolo.enabled = false;

        if (infoText) infoText.text = yoloOn ? "Carregando YOLO11n..." : "Carregando MobileNetV2...";

        if (yoloOn) yolo.enabled = true;
        else mobileNet.enabled = true;

        if (buttonLabel)
            buttonLabel.text = yoloOn
                ? "Modelo: <b>YOLO11n</b>  <size=70%></size>"
                : "Modelo: <b>MobileNetV2</b>  <size=70%></size>";
    }

    void CreateButton()
    {
        Canvas canvas = infoText ? infoText.canvas.rootCanvas : FindFirstObjectByType<Canvas>();
        if (!canvas) { Debug.LogWarning("ModelSwitcher: nenhum Canvas encontrado"); return; }

        var go = new GameObject("BotaoModelo", typeof(RectTransform), typeof(Image), typeof(Button));
        var rt = (RectTransform)go.transform;
        rt.SetParent(canvas.transform, false);
        rt.anchorMin = rt.anchorMax = rt.pivot = new Vector2(1f, 1f);
        rt.anchoredPosition = new Vector2(-30f, -30f);
        rt.sizeDelta = new Vector2(460f, 70f);
        rt.SetAsLastSibling();

        var img = go.GetComponent<Image>();
        img.color = new Color(0.106f, 0.369f, 0.125f, 0.9f);

        var button = go.GetComponent<Button>();
        button.targetGraphic = img;
        var colors = button.colors;
        colors.highlightedColor = new Color(0.85f, 1f, 0.85f);
        colors.pressedColor = new Color(0.7f, 0.9f, 0.7f);
        button.colors = colors;
        button.onClick.AddListener(Toggle);

        var labelGo = new GameObject("Texto", typeof(RectTransform));
        var lrt = (RectTransform)labelGo.transform;
        lrt.SetParent(rt, false);
        lrt.anchorMin = Vector2.zero;
        lrt.anchorMax = Vector2.one;
        lrt.offsetMin = new Vector2(16f, 0f);
        lrt.offsetMax = new Vector2(-16f, 0f);

        buttonLabel = labelGo.AddComponent<TextMeshProUGUI>();
        buttonLabel.fontSize = 28;
        buttonLabel.color = Color.white;
        buttonLabel.alignment = TextAlignmentOptions.Center;
        buttonLabel.raycastTarget = false;
    }
}