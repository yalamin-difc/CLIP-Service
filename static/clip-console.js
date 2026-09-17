/* Multimodal AL Engine festival console -- vanilla JS only.
 *
 * This script NEVER creates or signs a service JWT, never reads
 * INTERNAL_JWT_SECRET or MONGODB_URI (they do not exist in the browser),
 * and never calls a protected CLIP endpoint (/items, /match,
 * /analyze-image, or the versioned engine-matching API) directly. It
 * reads the public /health/ready endpoint, and (P14) the public
 * /demo/status and /demo/match endpoints -- both served by
 * demo_router.py, which mints its own short-lived internal identity
 * server-side and never returns it here. This is the only network
 * traffic this script ever generates.
 */
(function () {
  "use strict";

  var STRINGS = {
    en: {
      title: "Multimodal AL Engine",
      subtitleLine1: "Multimodal Intelligence for Lost & Found",
      subtitleLine2: "Urban Intelligence Platform",
      festivalBanner: "Dubai AI Festival 2026 — Demonstration Environment",
      demoDataPill: "DEMO DATA",
      festivalStory:
        "The Urban Intelligence Platform uses multiple multimodal AI models, including CLIP and " +
        "SigLIP2, to convert visual and textual evidence into model-specific representations, " +
        "retrieve and rank relevant lost-and-found and asset-recovery candidates, and enrich " +
        "results with OCR, barcode, and contextual signals. CLIP and SigLIP2 operate in separate " +
        "embedding spaces -- their retrieval evidence feeds the Urban Intelligence workflow, " +
        "while final recovery decisions remain subject to governed backend rules and human " +
        "verification.",
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
      pStepSignals: "OCR + Barcode + contextual signals",
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
      abTitle: "AI Model Comparison",
      abUnavailableNotice:
        "Live side-by-side comparison is not enabled in this public demonstration console. " +
        "Protected CLIP/SigLIP2 operations require a signed internal service identity that can " +
        "never be issued to a browser. Ask the Urban Intelligence Backend team about a demo-safe " +
        "server-side facade for authorized demonstrators.",
      abLatency: "Latency",
      abTopCandidate: "Top candidate",
      abStatus: "Model status",
      abDisclaimer: "AI-assisted candidate retrieval — human verification required.",
      abCalibrationWarning:
        "SigLIP2 is an experimental, uncalibrated engine: its similarity scores are not directly " +
        "comparable to CLIP's scores or to each other, and neither engine's score is an ownership " +
        "probability. A higher number never means a more certain match.",
      orgTagline: "Urban Intelligence AI Lab",
      heroTitle: "Multimodal Asset Recovery AI",
      heroSubtitle: "Visual + multilingual intelligence for lost & found and urban asset recovery.",
      liveDemoTitle: "Live AI Demo",
      demoUnavailableNotice:
        "The live demo facade is not enabled in this environment. Ask the Urban Intelligence team " +
        "to set DEMO_FACADE_ENABLED=true on this canary to turn it on.",
      modeClip: "CLIP",
      modeSiglip2: "SigLIP2",
      modeCompare: "Compare",
      demoUploadLabel: "Drag & drop an image, or click to choose a file",
      demoTextLabel: "Optional description (English, Arabic, or mixed)",
      demoTextPlaceholder: "e.g. black suitcase with a red ribbon on the handle",
      analyzeButton: "Analyze Item",
      analyzingText: "Analyzing multimodal evidence…",
      demoEvidenceNote:
        "Ranked candidates below are retrieval evidence (OCR/barcode overlap, when available) -- " +
        "not a confirmed match.",
      calibrationTooltip:
        "Similarity scores are retrieval signals, not ownership probabilities. Scores from " +
        "different AI models are not directly comparable.",
      humanReviewBanner: "AI-assisted candidate retrieval — human verification required.",
      experimentalBadge: "Experimental / Uncalibrated",
      productionBaseline: "Production Baseline",
      experimentalUncalibrated: "Experimental / Uncalibrated",
      pStepInput: "Image / Text",
      pipelineIsolationNote:
        "CLIP 512-D and SigLIP2 1152-D embeddings are maintained and searched in separate " +
        "model-specific vector spaces.",
      gNoPii: "No PII required for model demonstration",
      gProdIsolated: "Production data inaccessible from demo facade",
      modelIdLabel: "Model",
      embeddingLabel: "Embedding",
      roleLabel: "Role",
      statusLabelShort: "Status",
      calibrationLabel: "Calibration",
      siglip2Role: "Advanced Multimodal Evaluation Engine",
      errorRateLimited: "Too many requests -- please wait a moment and try again.",
      errorEngineUnavailable: "This AI engine is temporarily unavailable. Please try again shortly.",
      errorTimeout: "The request took too long and timed out. Please try again.",
      errorInvalidRequest: "Please provide an image, a description, or both.",
      errorNetwork: "Could not reach the demo service. Check your connection and try again.",
      errorGeneric: "Something went wrong while analyzing this item. Please try again.",
      noImage: "No image",
      untitledItem: "Untitled item",
      similarityLabel: "Similarity",
      noCandidates: "No candidates found.",
      sameTop1Label: "Same Top-1",
      topKOverlapLabel: "Top-K overlap",
      latencyDiffLabel: "Latency difference",
      yes: "Yes",
      no: "No",
      readyLabel: "Ready",
      notReadyLabel: "Not ready",
    },
    ar: {
      title: "محرك الذكاء متعدد الوسائط",
      subtitleLine1: "ذكاء متعدد الوسائط للمفقودات والموجودات",
      subtitleLine2: "منصة الذكاء الحضري",
      festivalBanner: "مهرجان دبي للذكاء الاصطناعي 2026 — بيئة تجريبية",
      demoDataPill: "بيانات تجريبية",
      festivalStory:
        "تستخدم منصة الذكاء الحضري عدة نماذج ذكاء اصطناعي متعددة الوسائط، منها CLIP وSigLIP2، " +
        "لتحويل الأدلة البصرية والنصية إلى تمثيلات خاصة بكل نموذج، واسترجاع وترتيب المرشحين " +
        "ذوي الصلة بالمفقودات والموجودات واسترداد الأصول، وإثراء النتائج بإشارات التعرف الضوئي " +
        "والرموز الشريطية والسياق. يعمل كل من CLIP وSigLIP2 في فضاء تمثيل متجهي منفصل خاص به -- " +
        "تُستخدم أدلة الاسترجاع الخاصة بهما ضمن سير عمل الذكاء الحضري، بينما تبقى قرارات " +
        "الاسترداد النهائية خاضعة لقواعد الخلفية البرمجية المحوكمة وللمراجعة البشرية.",
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
      pStepSignals: "إشارات التعرف الضوئي والرمز الشريطي والسياق",
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
      abTitle: "مقارنة نماذج الذكاء الاصطناعي",
      abUnavailableNotice:
        "المقارنة المباشرة جنبًا إلى جنب غير مفعّلة في هذه اللوحة التجريبية العامة. تتطلب " +
        "عمليات CLIP/SigLIP2 المحمية هوية خدمة داخلية موقّعة لا يمكن إصدارها إلى المتصفح " +
        "مطلقًا. تواصل مع فريق الخلفية البرمجية للذكاء الحضري بخصوص واجهة عرض تجريبية آمنة " +
        "من جهة الخادم للعارضين المصرّح لهم.",
      abLatency: "زمن الاستجابة",
      abTopCandidate: "أفضل مرشح",
      abStatus: "حالة النموذج",
      abDisclaimer: "استرجاع مرشحين بمساعدة الذكاء الاصطناعي — يتطلب التحقق البشري.",
      abCalibrationWarning:
        "SigLIP2 محرك تجريبي غير معاير: درجات التشابه فيه غير قابلة للمقارنة المباشرة مع " +
        "درجات CLIP أو مع بعضها البعض، ولا تمثل درجة أي من المحركين احتمال ملكية. الدرجة " +
        "الأعلى لا تعني أبدًا مطابقة أكثر يقينًا.",
      orgTagline: "مختبر الذكاء الاصطناعي للذكاء الحضري",
      heroTitle: "الذكاء الاصطناعي متعدد الوسائط لاستعادة الأصول",
      heroSubtitle: "ذكاء بصري ومتعدد اللغات للمفقودات والموجودات واسترداد الأصول الحضرية.",
      liveDemoTitle: "عرض الذكاء الاصطناعي المباشر",
      demoUnavailableNotice:
        "واجهة العرض المباشر غير مفعّلة في هذه البيئة. تواصل مع فريق الذكاء الحضري لتفعيل " +
        "DEMO_FACADE_ENABLED=true على هذه النسخة التجريبية.",
      modeClip: "CLIP",
      modeSiglip2: "SigLIP2",
      modeCompare: "مقارنة",
      demoUploadLabel: "اسحب وأفلت صورة، أو انقر لاختيار ملف",
      demoTextLabel: "وصف اختياري (إنجليزي أو عربي أو مختلط)",
      demoTextPlaceholder: "مثال: حقيبة سفر سوداء عليها شريط أحمر على المقبض",
      analyzeButton: "تحليل العنصر",
      analyzingText: "جارٍ تحليل الأدلة متعددة الوسائط…",
      demoEvidenceNote:
        "المرشحون المرتبون أدناه هم أدلة استرجاع (تداخل التعرف الضوئي/الرمز الشريطي عند توفره) -- " +
        "وليسوا مطابقة مؤكدة.",
      calibrationTooltip:
        "درجات التشابه هي إشارات استرجاع وليست احتمالات ملكية. الدرجات من نماذج ذكاء اصطناعي " +
        "مختلفة غير قابلة للمقارنة المباشرة.",
      humanReviewBanner: "استرجاع مرشحين بمساعدة الذكاء الاصطناعي — يتطلب التحقق البشري.",
      experimentalBadge: "تجريبي / غير معاير",
      productionBaseline: "الأساس الإنتاجي",
      experimentalUncalibrated: "تجريبي / غير معاير",
      pStepInput: "صورة / نص",
      pipelineIsolationNote:
        "تُحفظ تمثيلات CLIP ذات 512 بُعدًا وSigLIP2 ذات 1152 بُعدًا وتُبحث في فضاءات متجهية " +
        "منفصلة خاصة بكل نموذج.",
      gNoPii: "لا حاجة لبيانات شخصية لعرض النموذج",
      gProdIsolated: "بيانات الإنتاج غير قابلة للوصول من واجهة العرض التجريبي",
      modelIdLabel: "النموذج",
      embeddingLabel: "التمثيل المتجهي",
      roleLabel: "الدور",
      statusLabelShort: "الحالة",
      calibrationLabel: "المعايرة",
      siglip2Role: "محرك تقييم متقدم متعدد الوسائط",
      errorRateLimited: "طلبات كثيرة جدًا -- يرجى الانتظار قليلاً والمحاولة مرة أخرى.",
      errorEngineUnavailable: "محرك الذكاء الاصطناعي هذا غير متاح مؤقتًا. يرجى المحاولة لاحقًا.",
      errorTimeout: "استغرق الطلب وقتًا طويلاً وانتهت مهلته. يرجى المحاولة مرة أخرى.",
      errorInvalidRequest: "يرجى تقديم صورة أو وصف أو كليهما.",
      errorNetwork: "تعذر الوصول إلى خدمة العرض التجريبي. تحقق من الاتصال وحاول مرة أخرى.",
      errorGeneric: "حدث خطأ ما أثناء تحليل هذا العنصر. يرجى المحاولة مرة أخرى.",
      noImage: "لا توجد صورة",
      untitledItem: "عنصر بدون عنوان",
      similarityLabel: "التشابه",
      noCandidates: "لم يتم العثور على مرشحين.",
      sameTop1Label: "نفس المرشح الأول",
      topKOverlapLabel: "تداخل أفضل النتائج",
      latencyDiffLabel: "فرق زمن الاستجابة",
      yes: "نعم",
      no: "لا",
      readyLabel: "جاهز",
      notReadyLabel: "غير جاهز",
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
    document.querySelectorAll("[data-i18n-placeholder]").forEach(function (el) {
      var key = el.getAttribute("data-i18n-placeholder");
      if (strings[key] != null) {
        el.setAttribute("placeholder", strings[key]);
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

  /* ---- P14 live demo facade -------------------------------------------
   * Talks ONLY to the public /demo/status and /demo/match endpoints below.
   * Never attaches a bearer credential header, never signs a token, and
   * never calls /match, /items, or any protected engine-matching endpoint
   * directly from the browser -- the server-side demo facade
   * (demo_router.py) is the only thing that ever holds a signed internal
   * identity, and it never returns one to us.
   */
  var demoState = { mode: "clip", file: null };

  var demoCard = document.getElementById("live-demo-card");
  var demoUnavailableNotice = document.getElementById("demo-unavailable-notice");
  var demoInteractive = document.getElementById("demo-interactive");
  var demoUploadInput = document.getElementById("demo-upload-input");
  var demoUploadDrop = document.getElementById("demo-upload-drop");
  var demoPreviewImg = document.getElementById("demo-preview-img");
  var demoPreviewEmpty = document.getElementById("demo-preview-empty");
  var demoTextInput = document.getElementById("demo-text-input");
  var demoAnalyzeBtn = document.getElementById("demo-analyze-btn");
  var demoProcessing = document.getElementById("demo-processing");
  var demoErrorBox = document.getElementById("demo-error");
  var demoResults = document.getElementById("demo-results");
  var demoResultsSingle = document.getElementById("demo-results-single");
  var demoResultsCompare = document.getElementById("demo-results-compare");
  var demoSingleCandidates = document.getElementById("demo-single-candidates");

  function currentLang() {
    return STRINGS[document.getElementById("doc-root").getAttribute("lang")] || STRINGS.en;
  }

  function updateAnalyzeButtonState() {
    if (!demoAnalyzeBtn) return;
    var hasText = !!(demoTextInput && demoTextInput.value && demoTextInput.value.trim().length > 0);
    demoAnalyzeBtn.disabled = !demoState.file && !hasText;
  }

  function setPreviewFile(file) {
    demoState.file = file || null;
    if (!file) {
      if (demoPreviewImg) {
        demoPreviewImg.hidden = true;
        demoPreviewImg.removeAttribute("src");
      }
      if (demoPreviewEmpty) demoPreviewEmpty.hidden = false;
      updateAnalyzeButtonState();
      return;
    }
    var url = URL.createObjectURL(file);
    if (demoPreviewImg) {
      demoPreviewImg.src = url;
      demoPreviewImg.hidden = false;
    }
    if (demoPreviewEmpty) demoPreviewEmpty.hidden = true;
    updateAnalyzeButtonState();
  }

  if (demoUploadInput) {
    demoUploadInput.addEventListener("change", function () {
      var file = demoUploadInput.files && demoUploadInput.files[0];
      setPreviewFile(file || null);
    });
  }

  if (demoUploadDrop) {
    ["dragenter", "dragover"].forEach(function (eventName) {
      demoUploadDrop.addEventListener(eventName, function (event) {
        event.preventDefault();
        demoUploadDrop.classList.add("is-dragover");
      });
    });
    ["dragleave", "drop"].forEach(function (eventName) {
      demoUploadDrop.addEventListener(eventName, function (event) {
        event.preventDefault();
        demoUploadDrop.classList.remove("is-dragover");
      });
    });
    demoUploadDrop.addEventListener("drop", function (event) {
      var transfer = event.dataTransfer;
      var file = transfer && transfer.files && transfer.files[0];
      if (!file) return;
      if (demoUploadInput) {
        try {
          var dataTransfer = new DataTransfer();
          dataTransfer.items.add(file);
          demoUploadInput.files = dataTransfer.files;
        } catch (err) {
          /* DataTransfer construction unsupported in this browser -- the
           * preview and upload below still use the dropped File object
           * directly via demoState.file, so drag/drop still works. */
        }
      }
      setPreviewFile(file);
    });
  }

  if (demoTextInput) {
    demoTextInput.addEventListener("input", updateAnalyzeButtonState);
  }

  var modeTabs = Array.prototype.slice.call(document.querySelectorAll(".mode-tab"));
  modeTabs.forEach(function (tab) {
    tab.addEventListener("click", function () {
      demoState.mode = tab.getAttribute("data-mode") || "clip";
      modeTabs.forEach(function (other) {
        var active = other === tab;
        other.classList.toggle("is-active", active);
        other.setAttribute("aria-selected", active ? "true" : "false");
      });
    });
  });

  function showDemoError(message) {
    if (!demoErrorBox) return;
    demoErrorBox.textContent = message;
    demoErrorBox.hidden = false;
  }

  function clearDemoError() {
    if (!demoErrorBox) return;
    demoErrorBox.hidden = true;
    demoErrorBox.textContent = "";
  }

  function errorMessageFor(status, lang) {
    if (status === 429) return lang.errorRateLimited;
    if (status === 503) return lang.errorEngineUnavailable;
    if (status === "timeout") return lang.errorTimeout;
    if (status === 400) return lang.errorInvalidRequest;
    if (status === "network") return lang.errorNetwork;
    return lang.errorGeneric;
  }

  function buildCandidateItem(candidate, lang) {
    var row = document.createElement("div");
    row.className = "candidate-item";

    if (candidate.imageUrl) {
      var img = document.createElement("img");
      img.className = "candidate-thumb";
      img.src = candidate.imageUrl;
      img.alt = candidate.title || "";
      row.appendChild(img);
    } else {
      var placeholder = document.createElement("div");
      placeholder.className = "candidate-thumb candidate-thumb--empty";
      placeholder.textContent = lang.noImage;
      row.appendChild(placeholder);
    }

    var meta = document.createElement("div");
    meta.className = "candidate-meta";

    var rank = document.createElement("span");
    rank.className = "candidate-rank";
    rank.textContent = "#" + candidate.rank;
    meta.appendChild(rank);

    var title = document.createElement("span");
    title.className = "candidate-title";
    title.textContent = candidate.title || lang.untitledItem;
    meta.appendChild(title);

    if (typeof candidate.score === "number") {
      var score = document.createElement("span");
      score.className = "candidate-score";
      score.textContent = lang.similarityLabel + ": " + candidate.score.toFixed(3);
      meta.appendChild(score);
    }

    if (candidate.explanation) {
      var reason = document.createElement("span");
      reason.className = "candidate-reason";
      reason.textContent = candidate.explanation;
      meta.appendChild(reason);
    }

    row.appendChild(meta);
    return row;
  }

  function renderCandidateList(container, candidates, lang) {
    if (!container) return;
    container.textContent = "";
    if (!candidates || candidates.length === 0) {
      var empty = document.createElement("p");
      empty.className = "muted";
      empty.textContent = lang.noCandidates;
      container.appendChild(empty);
      return;
    }
    candidates.forEach(function (candidate) {
      container.appendChild(buildCandidateItem(candidate, lang));
    });
  }

  function renderSingleResult(data) {
    var lang = currentLang();
    if (demoResultsCompare) demoResultsCompare.hidden = true;
    if (demoResultsSingle) demoResultsSingle.hidden = false;
    renderCandidateList(demoSingleCandidates, data.candidates, lang);
  }

  function renderEngineColumn(prefix, side, lang) {
    var statusEl = document.getElementById(prefix + "-status");
    var latencyEl = document.getElementById(prefix + "-latency");
    var listEl = document.getElementById(prefix + "-candidates");
    if (!side || side.status !== "success") {
      if (statusEl) statusEl.textContent = lang.unavailable;
      if (latencyEl) latencyEl.textContent = "—";
      renderCandidateList(listEl, [], lang);
      return;
    }
    if (statusEl) statusEl.textContent = lang.readyLabel;
    if (latencyEl) latencyEl.textContent = "~" + side.latencyMs + " ms";
    renderCandidateList(listEl, side.candidates, lang);
  }

  function renderCompareResult(data) {
    var lang = currentLang();
    if (demoResultsSingle) demoResultsSingle.hidden = true;
    if (demoResultsCompare) demoResultsCompare.hidden = false;
    renderEngineColumn("compare-clip", data.clip, lang);
    renderEngineColumn("compare-siglip2", data.siglip2, lang);

    var comparison = data.comparison || {};
    var sameTop1El = document.getElementById("compare-same-top1");
    var overlapEl = document.getElementById("compare-overlap");
    var latencyDiffEl = document.getElementById("compare-latency-diff");
    if (sameTop1El) {
      sameTop1El.textContent =
        lang.sameTop1Label + ": " + (comparison.sameTop1 === true ? lang.yes : comparison.sameTop1 === false ? lang.no : "—");
    }
    if (overlapEl) {
      overlapEl.textContent = lang.topKOverlapLabel + ": " + (comparison.topKOverlap != null ? comparison.topKOverlap : "—");
    }
    if (latencyDiffEl) {
      latencyDiffEl.textContent =
        lang.latencyDiffLabel + ": " + (comparison.latencyDifferenceMs != null ? comparison.latencyDifferenceMs + " ms" : "—");
    }
  }

  function runDemoAnalysis() {
    if (!demoAnalyzeBtn || demoAnalyzeBtn.disabled) return;
    var lang = currentLang();
    clearDemoError();
    demoAnalyzeBtn.disabled = true;
    if (demoProcessing) demoProcessing.hidden = false;
    if (demoResults && !demoResults.hidden) {
      demoResults.classList.add("is-stale");
    }

    var formData = new FormData();
    formData.append("mode", demoState.mode);
    if (demoState.file) {
      formData.append("file", demoState.file);
    }
    var textValue = demoTextInput ? demoTextInput.value.trim() : "";
    if (textValue) {
      formData.append("text", textValue);
    }

    var timedOut = false;
    var controller = typeof AbortController !== "undefined" ? new AbortController() : null;
    var timeoutId = controller
      ? setTimeout(function () {
          timedOut = true;
          controller.abort();
        }, 45000)
      : null;

    fetch("/demo/match", {
      method: "POST",
      body: formData,
      signal: controller ? controller.signal : undefined,
    })
      .then(function (response) {
        if (timeoutId) clearTimeout(timeoutId);
        if (!response.ok) {
          var error = new Error("demo_match_failed");
          error.status = response.status;
          throw error;
        }
        return response.json();
      })
      .then(function (data) {
        if (demoResults) {
          demoResults.hidden = false;
          demoResults.classList.remove("is-stale");
        }
        if (data.mode === "compare") {
          renderCompareResult(data);
        } else {
          renderSingleResult(data);
        }
      })
      .catch(function (error) {
        if (demoResults) demoResults.classList.remove("is-stale");
        if (timedOut) {
          showDemoError(errorMessageFor("timeout", lang));
        } else if (error && typeof error.status === "number") {
          showDemoError(errorMessageFor(error.status, lang));
        } else {
          showDemoError(errorMessageFor("network", lang));
        }
      })
      .finally(function () {
        if (demoProcessing) demoProcessing.hidden = true;
        updateAnalyzeButtonState();
      });
  }

  if (demoAnalyzeBtn) {
    demoAnalyzeBtn.addEventListener("click", runDemoAnalysis);
  }

  function setModelCard(key, ready, dimension) {
    var lang = currentLang();
    var statusEl = document.getElementById("model-card-" + key + "-status");
    var dimEl = document.getElementById("model-card-" + key + "-dim");
    if (statusEl) {
      if (ready === true) {
        statusEl.textContent = lang.readyLabel;
        statusEl.classList.add("is-ready");
      } else if (ready === false) {
        statusEl.textContent = lang.notReadyLabel;
        statusEl.classList.remove("is-ready");
      } else {
        statusEl.textContent = lang.unavailable;
        statusEl.classList.remove("is-ready");
      }
    }
    if (dimEl && typeof dimension === "number") {
      dimEl.textContent = dimension + "-D";
    }
  }

  function refreshDemoStatus() {
    fetch("/demo/status", { headers: { Accept: "application/json" } })
      .then(function (response) {
        if (!response.ok) {
          throw new Error("demo_disabled");
        }
        return response.json();
      })
      .then(function (data) {
        if (demoUnavailableNotice) demoUnavailableNotice.hidden = true;
        if (demoInteractive) demoInteractive.hidden = false;
        var clip = data.clip || {};
        var siglip2 = data.siglip2 || {};
        setModelCard("clip", clip.ready, clip.embeddingDimension);
        setModelCard("siglip2", siglip2.ready, siglip2.embeddingDimension);
      })
      .catch(function () {
        if (demoUnavailableNotice) demoUnavailableNotice.hidden = false;
        if (demoInteractive) demoInteractive.hidden = true;
        setModelCard("clip", null, null);
        setModelCard("siglip2", null, null);
      });
  }

  if (demoCard) {
    refreshDemoStatus();
    setInterval(refreshDemoStatus, 20000);
  }

  updateAnalyzeButtonState();
})();
