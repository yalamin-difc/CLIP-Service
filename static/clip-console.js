/* CLIP AI Engine festival console -- vanilla JS only.
 *
 * This script NEVER creates or signs a service JWT, never reads
 * INTERNAL_JWT_SECRET or MONGODB_URI (they do not exist in the browser),
 * and never calls a protected CLIP endpoint (/items, /match,
 * /analyze-image). It only reads the public /health/ready endpoint and
 * lets the visitor preview a local image file -- the preview never leaves
 * the browser, because there is no browser-safe endpoint to send it to.
 */
(function () {
  "use strict";

  var STRINGS = {
    en: {
      title: "CLIP AI Engine",
      subtitleLine1: "Multimodal Intelligence for Lost & Found",
      subtitleLine2: "Urban Intelligence Platform",
      festivalBanner: "Dubai AI Festival 2026 — Demonstration Environment",
      demoDataPill: "DEMO DATA",
      festivalStory:
        "CLIP is one intelligence layer of Al-Amen Technology's AI-powered Lost & Found / " +
        "Urban Intelligence Platform. It converts visual and textual evidence into comparable " +
        "representations, retrieves relevant candidates, and combines with OCR and barcode " +
        "signals. Final match decisions remain governed by the Urban Intelligence Backend and " +
        "human review.",
      serviceStatus: "Service Status",
      model: "Model",
      ocr: "OCR",
      storage: "Storage",
      device: "Device",
      ready: "Ready",
      degraded: "Degraded",
      unavailable: "Unavailable",
      statusSource: "Live values from the public readiness endpoint (/health/ready).",
      modelProvenance: "Model Provenance",
      modelId: "Model ID",
      modelRevision: "Model revision",
      embeddingDimension: "Embedding dimension",
      preprocessingVersion: "Preprocessing version",
      scoringVersion: "Scoring version",
      serviceVersion: "Service version",
      provenanceNote:
        "Model provenance is captured to ensure matching results can be traced to the AI " +
        "configuration used during inference.",
      pipelineTitle: "AI Pipeline",
      pStepImage: "Image",
      pStepPreprocess: "Preprocessing",
      pStepVector: "512-dimensional representation",
      pStepRetrieval: "Authorized corpus retrieval",
      pStepSignals: "OCR + barcode signals",
      pStepRanked: "Ranked candidates",
      pStepFusion: "Backend multimodal fusion",
      pStepReview: "Human review",
      imageAnalysis: "Image Analysis",
      analysisUnavailableNotice:
        "Live image analysis is not enabled in this public demonstration console. Protected " +
        "CLIP operations require a signed internal service identity that can never be issued " +
        "to a browser. See the API documentation below, or ask the Urban Intelligence Backend " +
        "team about a demo-safe server-side facade.",
      uploadLabel: "Upload Lost/Found Item Image",
      previewPlaceholder: "No image selected",
      detectedText: "Detected Text",
      detectedBarcode: "Detected Barcode",
      matchingCandidates: "Matching Candidates",
      similarity: "Similarity",
      similarityDisclaimer:
        "Similarity values represent retrieval evidence. Final match decisions are made by " +
        "the Urban Intelligence Backend using multimodal evidence and governed decision rules.",
      trustedAi: "Trusted AI",
      gTenant: "Tenant isolated",
      gSite: "Site scoped",
      gAuth: "Signed service authentication",
      gTrace: "Request traceability",
      gProvenance: "Model provenance",
      gDemo: "Synthetic demo-data separation",
      gEmbeddings: "No raw embeddings exposed",
      gHuman: "Human-in-the-loop decision",
      apiDocumentation: "API Documentation",
      openSwagger: "Open Swagger",
      openApiSpec: "OpenAPI Specification",
      systemReadiness: "System Readiness",
      footerNote: "Demonstration console — synthetic festival data only.",
    },
    ar: {
      title: "محرك CLIP للذكاء الاصطناعي",
      subtitleLine1: "ذكاء متعدد الوسائط للمفقودات والموجودات",
      subtitleLine2: "منصة الذكاء الحضري",
      festivalBanner: "مهرجان دبي للذكاء الاصطناعي 2026 — بيئة تجريبية",
      demoDataPill: "بيانات تجريبية",
      festivalStory:
        "يُعد CLIP إحدى طبقات الذكاء في منصة الذكاء الحضري / المفقودات والموجودات التابعة " +
        "لشركة الأمين للتقنية. يحوّل الأدلة البصرية والنصية إلى تمثيلات قابلة للمقارنة، " +
        "ويسترجع المرشحين ذوي الصلة، ويجمعها مع إشارات التعرف الضوئي على الحروف والرموز " +
        "الشريطية. تبقى قرارات المطابقة النهائية خاضعة لحوكمة الخلفية البرمجية للذكاء " +
        "الحضري وللمراجعة البشرية.",
      serviceStatus: "حالة الخدمة",
      model: "النموذج",
      ocr: "التعرف الضوئي على الحروف",
      storage: "التخزين",
      device: "الجهاز",
      ready: "جاهز",
      degraded: "متدهور",
      unavailable: "غير متاح",
      statusSource: "قيم مباشرة من نقطة الجاهزية العامة (/health/ready).",
      modelProvenance: "مصدر النموذج",
      modelId: "معرّف النموذج",
      modelRevision: "إصدار النموذج",
      embeddingDimension: "بُعد التمثيل المتجهي",
      preprocessingVersion: "إصدار المعالجة المسبقة",
      scoringVersion: "إصدار التقييم",
      serviceVersion: "إصدار الخدمة",
      provenanceNote:
        "يتم تسجيل مصدر النموذج لضمان إمكانية تتبع نتائج المطابقة إلى تهيئة الذكاء " +
        "الاصطناعي المستخدمة أثناء الاستدلال.",
      pipelineTitle: "خط أنابيب الذكاء الاصطناعي",
      pStepImage: "الصورة",
      pStepPreprocess: "المعالجة المسبقة",
      pStepVector: "تمثيل متجهي بـ 512 بُعدًا",
      pStepRetrieval: "استرجاع من المجموعة المصرّح بها",
      pStepSignals: "إشارات التعرف الضوئي والرمز الشريطي",
      pStepRanked: "مرشحون مرتّبون",
      pStepFusion: "دمج متعدد الوسائط في الخلفية البرمجية",
      pStepReview: "مراجعة بشرية",
      imageAnalysis: "تحليل الصورة",
      analysisUnavailableNotice:
        "تحليل الصور المباشر غير مفعّل في هذه اللوحة التجريبية العامة. تتطلب عمليات CLIP " +
        "المحمية هوية خدمة داخلية موقّعة لا يمكن إصدارها إلى المتصفح مطلقًا. راجع وثائق " +
        "واجهة البرمجة أدناه، أو تواصل مع فريق الخلفية البرمجية للذكاء الحضري بخصوص واجهة " +
        "عرض تجريبية آمنة من جهة الخادم.",
      uploadLabel: "تحميل صورة عنصر مفقود/موجود",
      previewPlaceholder: "لم يتم اختيار صورة",
      detectedText: "النص المكتشف",
      detectedBarcode: "الرمز الشريطي المكتشف",
      matchingCandidates: "المرشحون المطابقون",
      similarity: "درجة التشابه",
      similarityDisclaimer:
        "تمثل قيم التشابه أدلة الاسترجاع. تُتخذ قرارات المطابقة النهائية من قبل الخلفية " +
        "البرمجية للذكاء الحضري باستخدام أدلة متعددة الوسائط وقواعد حوكمة معتمدة.",
      trustedAi: "ذكاء اصطناعي موثوق",
      gTenant: "معزول على مستوى المستأجر",
      gSite: "محدد النطاق حسب الموقع",
      gAuth: "مصادقة خدمة موقّعة",
      gTrace: "إمكانية تتبع الطلبات",
      gProvenance: "مصدر النموذج",
      gDemo: "فصل بيانات العرض التجريبي التركيبية",
      gEmbeddings: "لا يتم كشف أي تمثيلات متجهية خام",
      gHuman: "قرار بمراجعة بشرية",
      apiDocumentation: "وثائق واجهة البرمجة",
      openSwagger: "فتح Swagger",
      openApiSpec: "مواصفات OpenAPI",
      systemReadiness: "جاهزية النظام",
      footerNote: "لوحة تجريبية — بيانات مهرجان تركيبية فقط.",
    },
  };

  function applyLanguage(lang) {
    var strings = STRINGS[lang] || STRINGS.en;
    document.querySelectorAll("[data-i18n]").forEach(function (el) {
      var key = el.getAttribute("data-i18n");
      if (strings[key] != null) {
        el.textContent = strings[key];
      }
    });
    var root = document.getElementById("doc-root");
    if (lang === "ar") {
      root.setAttribute("lang", "ar");
      root.setAttribute("dir", "rtl");
    } else {
      root.setAttribute("lang", "en");
      root.setAttribute("dir", "ltr");
    }
    document.getElementById("lang-en").classList.toggle("is-active", lang === "en");
    document.getElementById("lang-ar").classList.toggle("is-active", lang === "ar");
    try {
      window.localStorage.setItem("clipConsoleLang", lang);
    } catch (err) {
      /* private-mode/blocked storage -- language toggle still works this session */
    }
  }

  document.getElementById("lang-en").addEventListener("click", function () {
    applyLanguage("en");
  });
  document.getElementById("lang-ar").addEventListener("click", function () {
    applyLanguage("ar");
  });

  var initialLang = "en";
  try {
    initialLang = window.localStorage.getItem("clipConsoleLang") || "en";
  } catch (err) {
    initialLang = "en";
  }
  applyLanguage(initialLang);

  function setStatusState(state, label) {
    var dot = document.getElementById("status-dot");
    dot.setAttribute("data-state", state);
    document.getElementById("status-label").textContent = label;
  }

  function checkLabel(value) {
    var lang = STRINGS[document.getElementById("doc-root").getAttribute("lang")] || STRINGS.en;
    if (value === true) return lang.ready;
    if (value === false) return lang.unavailable;
    return lang.unavailable;
  }

  function refreshReadiness() {
    var lang = STRINGS[document.getElementById("doc-root").getAttribute("lang")] || STRINGS.en;
    fetch("/health/ready", { headers: { Accept: "application/json" } })
      .then(function (response) {
        return response.json().then(function (body) {
          return { ok: response.ok, body: body };
        });
      })
      .then(function (result) {
        var checks = (result.body && result.body.checks) || {};
        var overallReady = result.ok && result.body && result.body.status === "ready";
        var anyCoreCheck =
          typeof checks.modelLoaded === "boolean" || typeof checks.databaseReady === "boolean";
        if (overallReady) {
          setStatusState("ready", lang.ready);
        } else if (anyCoreCheck) {
          setStatusState("degraded", lang.degraded);
        } else {
          setStatusState("unavailable", lang.unavailable);
        }
        document.getElementById("status-model").textContent = checkLabel(checks.modelLoaded);
        document.getElementById("status-ocr").textContent = checkLabel(checks.ocrDependencyReady);
        document.getElementById("status-storage").textContent = checkLabel(checks.databaseReady);
        document.getElementById("status-device").textContent =
          typeof checks.device === "string" && checks.device ? checks.device : lang.unavailable;
      })
      .catch(function () {
        setStatusState("unavailable", lang.unavailable);
        ["status-model", "status-ocr", "status-storage", "status-device"].forEach(function (id) {
          document.getElementById(id).textContent = lang.unavailable;
        });
      });
  }

  refreshReadiness();
  setInterval(refreshReadiness, 15000);

  var uploadInput = document.getElementById("upload-input");
  var previewImg = document.getElementById("preview-img");
  var previewEmpty = document.getElementById("preview-empty");
  if (uploadInput) {
    uploadInput.addEventListener("change", function () {
      var file = uploadInput.files && uploadInput.files[0];
      if (!file) {
        previewImg.hidden = true;
        previewEmpty.hidden = false;
        return;
      }
      var url = URL.createObjectURL(file);
      previewImg.src = url;
      previewImg.hidden = false;
      previewEmpty.hidden = true;
    });
  }
})();
