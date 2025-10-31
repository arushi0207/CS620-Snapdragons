import React, { useMemo, useState } from "react";
import axios from "axios";
import BottomNav from "./components/bottomnavbar";
import type { LucideIcon } from "lucide-react";
import {
  Target,
  Upload,
  Eye,
  PersonStanding,
  Smile,
  Menu,
  CheckCircle,
  ArrowRight,
  TrendingUp,
  Hand,
  Move,
} from "lucide-react";

// -------- Types --------
type Step = 1 | 2 | 3 | 4 | 5;
type TailwindBrand = "indigo" | "yellow" | "red" | "green";

// -------- Reusable Components --------
type HeaderProps = { onNavigate: (step: Step) => void; currentStep: Step };

const Header: React.FC<HeaderProps> = ({ onNavigate, currentStep }) => (
  <header className="flex justify-between items-center p-4 bg-white border-b border-gray-100 shadow-sm sticky top-0 z-10">
    <h1 className="text-2xl font-extrabold text-indigo-600 tracking-tight">
      Speak<span className="text-gray-900">Easy</span> AI Coach
    </h1>
    <nav className="hidden sm:flex space-x-4">
      <NavItem title="Home" step={1} currentStep={currentStep} onClick={() => onNavigate(1)} />
      <NavItem title="Goals" step={2} currentStep={currentStep} onClick={() => onNavigate(2)} />
      <NavItem title="Upload" step={3} currentStep={currentStep} onClick={() => onNavigate(3)} />
      <NavItem title="Review" step={4} currentStep={currentStep} onClick={() => onNavigate(4)} />
      <NavItem title="Analytics" step={5} currentStep={currentStep} onClick={() => onNavigate(5)} />
    </nav>
    <Menu className="sm:hidden text-indigo-600 cursor-pointer" size={24} />
  </header>
);

type NavItemProps = { title: string; step: Step; currentStep: Step; onClick: () => void };

const NavItem: React.FC<NavItemProps> = ({ title, step, currentStep, onClick }) => (
  <button
    onClick={onClick}
    className={`px-3 py-1 text-sm font-medium rounded-full transition duration-150 ${
      currentStep === step
        ? "bg-indigo-600 text-white shadow-md"
        : "text-gray-600 hover:text-indigo-600 hover:bg-indigo-50"
    }`}
  >
    {title}
  </button>
);

type MetricCardProps = {
  title: string;
  value: string;
  icon: LucideIcon;
  color: TailwindBrand;
  description: string;
  highlight: boolean;
};

const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  icon: Icon,
  color,
  description,
  highlight,
}) => (
  <div className="bg-white p-6 rounded-xl shadow-lg border border-gray-50 hover:shadow-xl transition duration-300 transform hover:scale-[1.02]">
    <div className={`flex items-center space-x-4 mb-3 text-${color}-600`}>
      <Icon size={32} className={`bg-${color}-100 p-2 rounded-lg`} />
      <h3 className="text-lg font-semibold text-gray-800">{title}</h3>
    </div>
    <p className={`text-4xl font-extrabold mb-2 ${highlight ? `text-${color}-600` : "text-gray-900"}`}>
      {value}
    </p>
    <p className="text-sm text-gray-500">{description}</p>
  </div>
);

// -------- Screen Components --------
type WelcomeScreenProps = { onStart: () => void };
const WelcomeScreen: React.FC<WelcomeScreenProps> = ({ onStart }) => (
  <div className="text-center p-8">
    <Target className="w-16 h-16 mx-auto text-indigo-600 mb-4" />
    <h2 className="text-4xl font-extrabold text-gray-900 mb-4">SpeakEasy</h2>
    <p className="text-lg text-gray-600 mb-8 max-w-lg mx-auto">
      Your AI-powered coach for mastering interviews, presentations, and building career-ready confidence. Get clear, actionable feedback on what you say and <strong>how you present yourself</strong>.
    </p>
    <button
      onClick={onStart}
      className="px-8 py-3 bg-indigo-600 text-white font-semibold rounded-full shadow-lg hover:bg-indigo-700 transition duration-300 transform hover:scale-105"
    >
      Start My First Session
    </button>
  </div>
);

type GoalSettingScreenProps = { onNext: () => void };
const GoalSettingScreen: React.FC<GoalSettingScreenProps> = ({ onNext }) => {
  const [selectedGoal, setSelectedGoal] = useState<string>("Job Interview Prep");
  const goals = [
    { id: "interview", name: "Job Interview Prep", description: "Practice the STAR method and common Q&A." },
    { id: "presentation", name: "Class Presentation", description: "Improve structure, pace, and engagement." },
    { id: "pitch", name: "Networking Pitch", description: "Refine your elevator pitch for quick impact." },
    { id: "confidence", name: "General Confidence", description: "Reduce anxiety and improve overall delivery." },
  ] as const;

  return (
    <div className="max-w-3xl mx-auto p-6">
      <h2 className="text-3xl font-bold text-gray-900 mb-2">1. Set Your Practice Goal</h2>
      <p className="text-gray-500 mb-8">Tell us what you're working on today to get targeted feedback.</p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
        {goals.map((goal) => (
          <div
            key={goal.id}
            onClick={() => setSelectedGoal(goal.name)}
            className={`p-5 rounded-xl border-2 cursor-pointer transition duration-200 ${
              selectedGoal === goal.name ? "border-indigo-600 bg-indigo-50 shadow-md" : "border-gray-200 hover:border-indigo-300 bg-white"
            }`}
          >
            <div className="flex justify-between items-center">
              <p className="text-lg font-semibold text-gray-800">{goal.name}</p>
              {selectedGoal === goal.name && <CheckCircle size={20} className="text-indigo-600" />}
            </div>
            <p className="text-sm text-gray-500 mt-1">{goal.description}</p>
          </div>
        ))}
      </div>

      <button
        onClick={onNext}
        className="w-full flex items-center justify-center space-x-2 px-8 py-3 bg-indigo-600 text-white font-semibold rounded-xl shadow-md hover:bg-indigo-700 transition duration-300"
      >
        <span>Continue to Upload ({selectedGoal})</span>
        <ArrowRight size={20} />
      </button>
    </div>
  );
};

type UploadScreenProps = { onAnalysisComplete: () => void };
const UploadScreen: React.FC<UploadScreenProps> = ({ onAnalysisComplete }) => {
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [progress, setProgress] = useState<number>(0);
  const [annotatedUrl, setAnnotatedUrl] = useState<string>("");
  const [error, setError] = useState<string>("");

  const API_BASE = (import.meta as any).env?.VITE_API_BASE_URL || "http://localhost:8000";

  const onFileSelected: React.ChangeEventHandler<HTMLInputElement> = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    setError("");
    setProgress(0);
    setAnnotatedUrl("");

    try {
      const form = new FormData();
      form.append("video", file);

      const resp = await axios.post(`${API_BASE}/process-video`, form, {
        responseType: "blob",
        headers: { "Content-Type": "multipart/form-data" },
        onUploadProgress: (pe) => {
          if (!pe.total) return;
          const p = Math.round((pe.loaded / pe.total) * 100);
          setProgress(p);
        },
      });

      const blob = new Blob([resp.data], { type: "video/mp4" });
      const url = URL.createObjectURL(blob);
      setAnnotatedUrl(url);
      setProgress(100);
    } catch (err: any) {
      setError(err?.message || "Upload failed");
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-6 text-center">
      <h2 className="text-3xl font-bold text-gray-900 mb-2">2. Upload Your Practice Video</h2>
      <p className="text-gray-500 mb-8">
        Record your speech or mock interview on your phone, then upload the file here for AI analysis.
      </p>

      <div
        className={`border-4 border-dashed rounded-2xl p-12 mb-8 transition duration-300 ${
          isUploading ? "border-indigo-400 bg-indigo-50" : "border-gray-300 hover:border-indigo-500 hover:bg-gray-50"
        }`}
      >
        <Upload className="w-12 h-12 mx-auto text-indigo-500 mb-3" />
        <p className="text-gray-600 font-medium">Drag & drop your video here, or click to select file.</p>
        <p className="text-sm text-gray-400 mt-1">MP4 or MOV files under 500MB recommended.</p>
      </div>

      <input type="file" id="video-upload" accept="video/*" className="hidden" onChange={onFileSelected} />

      {!isUploading && (
        <label
          htmlFor="video-upload"
          className="inline-flex items-center space-x-2 px-8 py-3 bg-indigo-600 text-white font-semibold rounded-xl shadow-md cursor-pointer hover:bg-indigo-700 transition duration-300"
        >
          <Upload size={20} />
          <span>Select Video File</span>
        </label>
      )}

      {isUploading && (
        <div className="mt-8">
          <p className="text-lg font-medium text-gray-700 mb-2">Uploading... ({progress}%)</p>
          <div className="w-full bg-gray-200 rounded-full h-3">
            <div className="bg-indigo-600 h-3 rounded-full transition-all duration-500 ease-out" style={{ width: `${progress}%` }} />
          </div>
        </div>
      )}

      {!!error && <p className="text-red-600 mt-3 font-semibold">{error}</p>}

      {annotatedUrl && (
        <div className="mt-8 space-y-4">
          <p className="text-green-700 font-semibold">Analysis complete! Your annotated video is ready.</p>

          <div className="flex items-center justify-center gap-3">
            <a href={annotatedUrl} download="annotated.mp4" className="px-5 py-3 bg-gray-900 text-white rounded-xl shadow hover:bg-gray-700">
              Download MP4
            </a>
            <button
              onClick={onAnalysisComplete}
              className="px-5 py-3 bg-indigo-600 text-white rounded-xl shadow hover:bg-indigo-700"
            >
              Continue to Review
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

type ReviewScreenProps = { onNext: () => void };
const ReviewScreen: React.FC<ReviewScreenProps> = ({ onNext }) => {
  const analysisData = useMemo(
    () => ({
      handMovements: {
        value: "Frequent",
        color: "yellow" as TailwindBrand,
        desc: "Hands are active and sometimes distracting. Anchor gestures to key points.",
        icon: Hand as LucideIcon,
        highlight: true,
      },
      legMovements: {
        value: "Restless",
        color: "red" as TailwindBrand,
        desc: "Notable shifting/foot tapping detected. Aim for a planted, stable stance.",
        icon: Move as LucideIcon,
        highlight: true,
      },
      eyeGaze: {
        value: "92%",
        color: "green" as TailwindBrand,
        desc: "Excellent. You maintained strong eye contact with the camera.",
        icon: Eye as LucideIcon,
        highlight: false,
      },
      posture: {
        value: "Good",
        color: "green" as TailwindBrand,
        desc: "Confident and open stance maintained throughout the speech.",
        icon: PersonStanding as LucideIcon,
        highlight: false,
      },
      facialExpression: {
        rating: 4,
        comment:
          "Your energy and smile were engaging, especially when discussing your previous work. Use expression to emphasize key points.",
      },
    }),
    []
  );

  return (
    <div className="max-w-5xl mx-auto p-6">
      <h2 className="text-3xl font-bold text-gray-900 mb-2">3. Session Review: Job Interview Prep</h2>
      <p className="text-indigo-600 font-semibold mb-8">AI Analysis Complete! Score: 78/100</p>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-10">
        <MetricCard title="Hand Movements" value={analysisData.handMovements.value} icon={analysisData.handMovements.icon} color={analysisData.handMovements.color} description={analysisData.handMovements.desc} highlight={analysisData.handMovements.highlight} />
        <MetricCard title="Leg Movements" value={analysisData.legMovements.value} icon={analysisData.legMovements.icon} color={analysisData.legMovements.color} description={analysisData.legMovements.desc} highlight={analysisData.legMovements.highlight} />
        <MetricCard title="Eye Gaze" value={analysisData.eyeGaze.value} icon={analysisData.eyeGaze.icon} color={analysisData.eyeGaze.color} description={analysisData.eyeGaze.desc} highlight={analysisData.eyeGaze.highlight} />
        <MetricCard title="Posture" value={analysisData.posture.value} icon={analysisData.posture.icon} color={analysisData.posture.color} description={analysisData.posture.desc} highlight={analysisData.posture.highlight} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1 bg-white p-6 rounded-xl shadow-lg border border-gray-50">
          <div className="flex items-center space-x-3 text-indigo-600 mb-4">
            <Smile size={24} />
            <h3 className="text-xl font-semibold text-gray-800">Facial Expression</h3>
          </div>
          <p className="text-3xl font-extrabold text-yellow-600 mb-2">
            {"⭐".repeat(analysisData.facialExpression.rating)}
            {"☆".repeat(5 - analysisData.facialExpression.rating)}
          </p>
          <p className="text-gray-700">{analysisData.facialExpression.comment}</p>
        </div>

        <div className="lg:col-span-2 bg-indigo-50 p-6 rounded-xl shadow-inner border border-indigo-100">
          <div className="flex items-center space-x-3 text-indigo-600 mb-4">
            <TrendingUp size={24} />
            <h3 className="text-xl font-bold text-indigo-700">AI Coach Summary</h3>
          </div>
          <p className="text-gray-800 leading-relaxed">
            Strong confidence markers in <strong>Eye Gaze</strong> and <strong>Posture</strong>. Focus next on
            <strong> Hand Movements</strong> and <strong>Leg Movements</strong>: keep gestures purposeful and plant your stance to reduce lower-body fidgeting.
          </p>
        </div>
      </div>

      <div className="mt-10 flex justify-center">
        <button
          onClick={onNext}
          className="px-8 py-3 bg-gray-900 text-white font-semibold rounded-full shadow-lg hover:bg-gray-700 transition duration-300 transform hover:scale-105"
        >
          View Full Progress Dashboard
        </button>
      </div>
    </div>
  );
};

type AnalyticsScreenProps = { onNext: (s: Step) => void };
const AnalyticsScreen: React.FC<AnalyticsScreenProps> = ({ onNext }) => (
  <div className="max-w-5xl mx-auto p-6">
    <h2 className="text-3xl font-bold text-gray-900 mb-8">4. Progress Dashboard</h2>
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div className="bg-white p-6 rounded-xl shadow-lg border border-gray-50">
        <h3 className="text-xl font-semibold text-gray-800 mb-4 flex items-center space-x-2">
          <Hand size={20} className="text-indigo-600" /> <span>Hand Movement Control</span>
        </h3>
        <div className="h-40 bg-gray-100 rounded-lg flex items-end p-2 text-sm text-gray-500 relative">
          <div className="w-1/5 h-4/5 bg-indigo-200 rounded-t-sm mx-1" title="Week 1: Frequent" />
          <div className="w-1/5 h-3/5 bg-indigo-300 rounded-t-sm mx-1" title="Week 2: Moderate" />
          <div className="w-1/5 h-2/5 bg-indigo-400 rounded-t-sm mx-1" title="Week 3: Controlled" />
          <div className="w-1/5 h-2/5 bg-indigo-500 rounded-t-sm mx-1" title="Week 4: Controlled" />
          <p className="absolute bottom-2 left-2">Mock Progress Chart showing gestures becoming more purposeful.</p>
        </div>
        <p className="mt-4 text-sm text-gray-600">Goal: Maintain purposeful gestures aligned to key points.</p>
      </div>

      <div className="bg-white p-6 rounded-xl shadow-lg border border-gray-50">
        <h3 className="text-xl font-semibold text-gray-800 mb-4 flex items-center space-x-2">
          <Move size={20} className="text-indigo-600" /> <span>Leg Movement Stillness</span>
        </h3>
        <div className="h-40 bg-gray-100 rounded-lg flex items-start p-2 text-sm text-gray-500 relative">
          <div className="w-1/5 h-full bg-red-400 rounded-b-sm mx-1" style={{ height: "90%" }} title="Week 1: High fidgeting" />
          <div className="w-1/5 h-full bg-red-300 rounded-b-sm mx-1" style={{ height: "60%" }} title="Week 2: Moderate" />
          <div className="w-1/5 h-full bg-red-200 rounded-b-sm mx-1" style={{ height: "35%" }} title="Week 3: Low" />
          <div className="w-1/5 h-full bg-green-400 rounded-b-sm mx-1" style={{ height: "20%" }} title="Week 4: Stable (Goal)" />
          <p className="absolute top-2 left-2">Mock Progress Chart showing reduction in lower-body fidgeting.</p>
        </div>
        <p className="mt-4 text-sm text-gray-600">Goal: Keep feet planted and minimize shifting.</p>
      </div>
    </div>
    <div className="mt-10 text-center">
      <p className="text-lg text-gray-700 font-medium">Ready for your next session?</p>
      <button
        onClick={() => onNext(2)}
        className="mt-4 px-8 py-3 bg-indigo-600 text-white font-semibold rounded-full shadow-lg hover:bg-indigo-700 transition duration-300 transform hover:scale-105"
      >
        Start New Practice
      </button>
    </div>
  </div>
);

// -------- Main App Component --------
const App: React.FC = () => {
  const [step, setStep] = useState<Step>(1);

  const navigateTo = (newStep: Step) => {
    if (newStep === 4 && step < 3) return;
    setStep(newStep);
    window.scrollTo(0, 0);
  };

  const renderContent = () => {
    switch (step) {
      case 1:
        return <WelcomeScreen onStart={() => navigateTo(2)} />;
      case 2:
        return <GoalSettingScreen onNext={() => navigateTo(3)} />;
      case 3:
        return <UploadScreen onAnalysisComplete={() => navigateTo(4)} />;
      case 4:
        return <ReviewScreen onNext={() => navigateTo(5)} />;
      case 5:
        return <AnalyticsScreen onNext={navigateTo} />;
      default:
        return <WelcomeScreen onStart={() => navigateTo(2)} />;
    }
  };

  return (
    <div className="min-h-dvh bg-gray-50 font-sans antialiased flex flex-col">
      <Header onNavigate={navigateTo} currentStep={step} />
      <main className="flex-1 py-10 pb-24">
        <div className="mx-auto px-4 sm:px-6 lg:px-8 h-full">
          
          <div className="bg-white p-8 sm:p-12 rounded-2xl shadow-2xl
                          min-h-[calc(100dvh-128px)] flex items-center justify-center">
            {renderContent()}
          </div>
        </div>
      </main>
  
      <BottomNav currentStep={step} onNavigate={navigateTo} />
    </div>
  );
};

export default App;
