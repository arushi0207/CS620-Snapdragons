import React, { useEffect, useState } from "react";
import axios from "axios";
import BottomNav from "./components/bottomnavbar";
import type { LucideIcon } from "lucide-react";
import { toast } from "sonner";

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
  Info,
  Sun,
  Moon,
} from "lucide-react";

/* ---------- Types ---------- */
type Step = 1 | 2 | 3 | 4 | 5;
type TailwindBrand = "indigo" | "yellow" | "red" | "green";
type FeedbackKey = "handMovements" | "legMovements" | "eyeGaze" | "posture";

/* ---------- Reusable Components ---------- */
type HeaderProps = {
  onNavigate: (step: Step) => void;
  currentStep: Step;
  isDarkMode: boolean;
  onToggleTheme: () => void;
};

const Header: React.FC<HeaderProps> = ({
  onNavigate,
  currentStep,
  isDarkMode,
  onToggleTheme,
}) => (
  <header
    className={`flex justify-between items-center p-4 border-b shadow-sm sticky top-0 z-10 transition-colors duration-300 ${
      isDarkMode ? "bg-slate-900 border-slate-800" : "bg-white border-gray-100"
    }`}
  >
    <h1
      className={`text-2xl font-extrabold tracking-tight ${
        isDarkMode ? "text-indigo-300" : "text-indigo-600"
      }`}
    >
      Speak
      <span className={isDarkMode ? "text-white" : "text-gray-900"}>Easy</span>{" "}
      AI Coach
    </h1>

    <nav className="hidden sm:flex space-x-4">
      <NavItem
        title="Home"
        step={1}
        currentStep={currentStep}
        onClick={() => onNavigate(1)}
        isDarkMode={isDarkMode}
      />
      <NavItem
        title="Goals"
        step={2}
        currentStep={currentStep}
        onClick={() => onNavigate(2)}
        isDarkMode={isDarkMode}
      />
      <NavItem
        title="Upload"
        step={3}
        currentStep={currentStep}
        onClick={() => onNavigate(3)}
        isDarkMode={isDarkMode}
      />
      <NavItem
        title="Review"
        step={4}
        currentStep={currentStep}
        onClick={() => onNavigate(4)}
        isDarkMode={isDarkMode}
      />
      <NavItem
        title="Analytics"
        step={5}
        currentStep={currentStep}
        onClick={() => onNavigate(5)}
        isDarkMode={isDarkMode}
      />
    </nav>

    <div className="flex items-center gap-3">
      {/* Dark theme toggle */}
      <button
        type="button"
        onClick={onToggleTheme}
        className={`inline-flex items-center justify-center w-9 h-9 rounded-full border text-xs font-medium transition-colors duration-200 ${
          isDarkMode
            ? "bg-slate-800 border-slate-700 text-slate-100 hover:bg-slate-700"
            : "bg-white border-gray-200 text-gray-700 hover:bg-gray-50"
        }`}
        title="Toggle light / dark mode"
      >
        {isDarkMode ? <Sun size={16} /> : <Moon size={16} />}
      </button>

      <Menu
        className={`sm:hidden cursor-pointer ${
          isDarkMode ? "text-indigo-300" : "text-indigo-600"
        }`}
        size={24}
      />
    </div>
  </header>
);

type NavItemProps = {
  title: string;
  step: Step;
  currentStep: Step;
  onClick: () => void;
  isDarkMode: boolean;
};

const NavItem: React.FC<NavItemProps> = ({
  title,
  step,
  currentStep,
  onClick,
  isDarkMode,
}) => {
  const active =
    "bg-indigo-600 text-white shadow-md hover:bg-indigo-700 border border-indigo-500";
  const inactive = isDarkMode
    ? "text-slate-200 hover:text-white hover:bg-slate-800"
    : "text-gray-600 hover:text-indigo-600 hover:bg-indigo-50";

  return (
    <button
      onClick={onClick}
      className={`px-3 py-1 text-sm font-medium rounded-full transition duration-150 ${
        currentStep === step ? active : inactive
      }`}
    >
      {title}
    </button>
  );
};

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
    {/* NOTE: if you purge Tailwind, safelist these dynamic classes */}
    <div className={`flex items-center space-x-4 mb-3 text-${color}-600`}>
      <Icon size={32} className={`bg-${color}-100 p-2 rounded-lg`} />
      <h3 className="text-lg font-semibold text-gray-800">{title}</h3>
    </div>
    <p
      className={`text-4xl font-extrabold mb-2 ${
        highlight ? `text-${color}-600` : "text-gray-900"
      }`}
    >
      {value}
    </p>
    <p className="text-sm text-gray-500">{description}</p>
  </div>
);

/* ---------- Screens ---------- */

type WelcomeScreenProps = { onStart: () => void };

const WelcomeScreen: React.FC<WelcomeScreenProps> = ({ onStart }) => (
  <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-[1.4fr,1fr] gap-10 items-center py-10">
    {/* Left side: hero text + benefits */}
    <section className="space-y-6">
      <p className="text-xs font-semibold text-indigo-500 uppercase tracking-[0.2em]">
        SpeakEasy AI Coach
      </p>

      <h2 className="text-4xl md:text-5xl font-extrabold text-gray-900 leading-tight">
        Become the <span className="text-indigo-600">most confident speaker</span>{" "}
        in the room.
      </h2>

      <p className="text-base md:text-lg text-gray-600 max-w-xl">
        Practice interviews, class presentations, or elevator pitches with
        AI-powered feedback on your clarity, pacing, and body language.
      </p>

      {/* Benefits */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-4 flex flex-col gap-2 hover:shadow-md transition">
          <div className="inline-flex h-8 w-8 items-center justify-center rounded-full bg-indigo-50">
            <Target className="h-4 w-4 text-indigo-600" />
          </div>
          <p className="text-sm font-semibold text-gray-900">Goal-based practice</p>
          <p className="text-xs text-gray-500">
            Choose interview, presentation, or networking—and get targeted prompts.
          </p>
        </div>

        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-4 flex flex-col gap-2 hover:shadow-md transition">
          <div className="inline-flex h-8 w-8 items-center justify-center rounded-full bg-emerald-50">
            <TrendingUp className="h-4 w-4 text-emerald-600" />
          </div>
          <p className="text-sm font-semibold text-gray-900">Instant analytics</p>
          <p className="text-xs text-gray-500">
            See filler words, pacing, and confidence scores after every session.
          </p>
        </div>

        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-4 flex flex-col gap-2 hover:shadow-md transition">
          <div className="inline-flex h-8 w-8 items-center justify-center rounded-full bg-violet-50">
            <Smile className="h-4 w-4 text-violet-600" />
          </div>
          <p className="text-sm font-semibold text-gray-900">Safe practice space</p>
          <p className="text-xs text-gray-500">
            Rehearse as many times as you want before the real thing.
          </p>
        </div>
      </div>

      {/* CTA buttons */}
      <div className="flex flex-wrap items-center gap-3 pt-2">
        <button
          onClick={onStart}
          className="inline-flex items-center gap-2 rounded-full bg-indigo-600 px-6 py-3 text-sm md:text-base font-semibold text-white shadow-lg shadow-indigo-300/40 hover:bg-indigo-700 transition hover:scale-[1.02]"
        >
          Start my first session
          <ArrowRight className="h-4 w-4" />
        </button>
        <button className="text-sm text-gray-600 hover:text-gray-900 underline-offset-2 hover:underline">
          View sample feedback
        </button>
      </div>
    </section>

    {/* Right side: "Next session" card */}
    <aside className="bg-white rounded-3xl shadow-xl border border-gray-100 p-6 space-y-6">
      <div className="flex items-center justify-between gap-2">
        <div>
          <p className="text-xs font-medium text-indigo-500 uppercase tracking-[0.18em]">
            Next session
          </p>
          <p className="text-base md:text-lg font-semibold text-gray-900">
            Behavioral interview practice
          </p>
        </div>
        <span className="inline-flex items-center gap-1 rounded-full bg-indigo-50 px-3 py-1 text-xs font-medium text-indigo-700">
          <PersonStanding className="h-3 w-3" />
          10 min
        </span>
      </div>

      <div className="space-y-3 text-sm text-gray-600">
        <p className="font-medium text-gray-900">Today’s focus:</p>
        <ul className="space-y-1">
          <li className="flex gap-2">
            <CheckCircle className="h-4 w-4 mt-[2px] text-emerald-500" />
            <span>Answer “Tell me about yourself” with a strong hook.</span>
          </li>
          <li className="flex gap-2">
            <CheckCircle className="h-4 w-4 mt-[2px] text-emerald-500" />
            <span>Keep answers between 60–90 seconds.</span>
          </li>
          <li className="flex gap-2">
            <CheckCircle className="h-4 w-4 mt-[2px] text-emerald-500" />
            <span>Reduce filler words like “um” and “like”.</span>
          </li>
        </ul>
      </div>

      <div className="space-y-3">
        <p className="text-xs font-medium text-gray-500 uppercase tracking-[0.18em]">
          Last session snapshot
        </p>
        <div className="grid grid-cols-3 gap-3 text-center">
          <div className="rounded-2xl bg-gray-50 p-3">
            <p className="text-xs text-gray-500">Clarity</p>
            <p className="text-lg font-semibold text-gray-900">8.6</p>
          </div>
          <div className="rounded-2xl bg-gray-50 p-3">
            <p className="text-xs text-gray-500">Confidence</p>
            <p className="text-lg font-semibold text-gray-900">7.9</p>
          </div>
          <div className="rounded-2xl bg-gray-50 p-3">
            <p className="text-xs text-gray-500">Filler words</p>
            <p className="text-lg font-semibold text-emerald-600">-32%</p>
          </div>
        </div>
      </div>
    </aside>
  </div>
);

type GoalSettingScreenProps = { onNext: () => void };

const GOAL_STORAGE_KEY = "speakeasy:lastGoal";

const goals = [
  {
    id: "interview",
    name: "Job Interview Prep",
    description: "Practice the STAR method and common Q&A.",
    hoverText:
      "Best if you have an upcoming internship or job interview. Focuses on behavioral answers, STAR stories, and clear structure.",
  },
  {
    id: "presentation",
    name: "Class Presentation",
    description: "Improve structure, pace, and engagement.",
    hoverText:
      "Use this when preparing for a class or work presentation. Helps with flow, slide transitions, and audience engagement.",
  },
  {
    id: "pitch",
    name: "Networking Pitch",
    description: "Refine your elevator pitch for quick impact.",
    hoverText:
      "Perfect for coffee chats, career fairs, and networking events. Focuses on a tight 30–60 second intro about you.",
  },
  {
    id: "confidence",
    name: "General Confidence",
    description: "Reduce anxiety and improve overall delivery.",
    hoverText:
      "Choose this when you just want to get more comfortable speaking out loud—no specific event required.",
  },
] as const;

const GoalSettingScreen: React.FC<GoalSettingScreenProps> = ({ onNext }) => {
  const [selectedGoal, setSelectedGoal] = useState<string>(goals[0].name);
  const [recommendedGoal, setRecommendedGoal] = useState<string | null>(null);

  // Load last goal from localStorage on first render
  useEffect(() => {
    try {
      const lastGoal = localStorage.getItem(GOAL_STORAGE_KEY);
      if (lastGoal) {
        setSelectedGoal(lastGoal);
        setRecommendedGoal(lastGoal);
      }
    } catch (err) {
      console.error("Could not read last goal from storage", err);
    }
  }, []);

  const handleSelectGoal = (goalName: string) => {
    setSelectedGoal(goalName);
    try {
      localStorage.setItem(GOAL_STORAGE_KEY, goalName);
    } catch (err) {
      console.error("Could not save last goal to storage", err);
    }
  };

  const handleContinue = () => {
    try {
      localStorage.setItem(GOAL_STORAGE_KEY, selectedGoal);
    } catch (err) {
      console.error("Could not save last goal to storage", err);
    }
    onNext();
  };

  return (
    <div className="max-w-3xl mx-auto p-6">
      <h2 className="text-3xl font-bold text-gray-900 mb-2">
        1. Set Your Practice Goal
      </h2>
      <p className="text-gray-500 mb-2">
        Tell us what you're working on today to get targeted feedback.
      </p>

      {/* Short description under the goal options */}
      <div className="mb-6 rounded-lg bg-gray-50 border border-gray-200 px-4 py-3 text-sm text-gray-700 leading-relaxed">
        Each session type customizes your AI feedback style and focus. Choose
        the option that best matches your real-world speaking scenario—whether
        that’s interviews, presentations, networking situations, or improving
        overall confidence.
      </div>

      {/* Recommended banner based on previous session */}
      {recommendedGoal && (
        <div className="mb-6 rounded-lg bg-indigo-50 border border-indigo-100 px-4 py-3 text-sm text-indigo-800">
          Recommended based on your last session:
          <span className="font-semibold ml-1">{recommendedGoal}</span>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
        {goals.map((goal) => (
          <div
            key={goal.id}
            onClick={() => handleSelectGoal(goal.name)}
            className={`p-5 rounded-xl border-2 cursor-pointer transition duration-200 ${
              selectedGoal === goal.name
                ? "border-indigo-600 bg-indigo-50 shadow-md"
                : "border-gray-200 hover:border-indigo-300 bg-white"
            }`}
            title={goal.hoverText} // native browser tooltip on hover
          >
            <div className="flex justify-between items-center">
              <p className="text-lg font-semibold text-gray-800 flex items-center gap-2">
                {goal.name}
                {recommendedGoal === goal.name && (
                  <span className="text-xs px-2 py-0.5 rounded-full bg-indigo-100 text-indigo-700 font-semibold">
                    Recommended
                  </span>
                )}
              </p>

              {/* Info icon with hover explanation */}
              <button
                type="button"
                className="p-1 rounded-full border border-indigo-100 text-indigo-400 hover:bg-indigo-50 hover:text-indigo-700"
                onClick={(e) => e.stopPropagation()} // don’t toggle card on icon click
                title={goal.hoverText}
              >
                <Info size={14} />
              </button>
            </div>

            {/* Short description already present under each card */}
            <p className="text-sm text-gray-500 mt-1">{goal.description}</p>
          </div>
        ))}
      </div>

      <button
        onClick={handleContinue}
        className="w-full flex items-center justify-center space-x-2 px-8 py-3 bg-indigo-600 text-white font-semibold rounded-xl shadow-md hover:bg-indigo-700 transition duration-300"
      >
        <span>Continue to Upload ({selectedGoal})</span>
        <ArrowRight size={20} />
      </button>
    </div>
  );
};

/* ---------- Upload / Analysis Screen ---------- */

type UploadScreenProps = { onAnalysisComplete: () => void };

const UploadScreen: React.FC<UploadScreenProps> = ({ onAnalysisComplete }) => {
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [progress, setProgress] = useState<number>(0);
  const [annotatedUrl, setAnnotatedUrl] = useState<string>("");
  const [error, setError] = useState<string>("");

  // NEW for time remaining
  const [estimatedTime, setEstimatedTime] = useState<number | null>(null);

  const API_BASE =
    (import.meta as any).env?.VITE_API_BASE_URL || "http://localhost:8000";

  const onFileSelected: React.ChangeEventHandler<HTMLInputElement> = async (
    e
  ) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    setError("");
    setProgress(0);
    setAnnotatedUrl("");
    setEstimatedTime(null);

    const uploadStart = Date.now();

    try {
      const form = new FormData();
      form.append("video", file);

      const resp = await axios.post(`${API_BASE}/process-video`, form, {
        responseType: "blob",
        headers: { "Content-Type": "multipart/form-data" },
        onUploadProgress: (pe) => {
          if (!pe.total) return;

          const percent = Math.round((pe.loaded / pe.total) * 100);
          setProgress(percent);

          // estimate time remaining
          const elapsed = (Date.now() - uploadStart) / 1000; // seconds
          const rate = pe.loaded / elapsed; // bytes per sec
          const remainingBytes = pe.total - pe.loaded;
          const remainingSeconds = Math.ceil(remainingBytes / rate);
          if (Number.isFinite(remainingSeconds)) {
            setEstimatedTime(remainingSeconds);
          }
        },
      });

      const blob = new Blob([resp.data], { type: "video/mp4" });
      const url = URL.createObjectURL(blob);
      setAnnotatedUrl(url);
      setProgress(100);
      setEstimatedTime(0);
      toast.success("Video processed!");
    } catch (err: any) {
      const msg = err?.message || "Upload failed";
      setError(msg);
      toast.error(msg);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-6 text-center">
      <h2 className="text-3xl font-bold text-gray-900 mb-2">
        2. Upload Your Practice Video
      </h2>
      <p className="text-gray-500 mb-8">
        Record your speech or mock interview on your phone, then upload the file
        here for AI analysis.
      </p>

      <div
        className={`border-4 border-dashed rounded-2xl p-12 mb-8 transition duration-300 ${
          isUploading
            ? "border-indigo-400 bg-indigo-50"
            : "border-gray-300 hover:border-indigo-500 hover:bg-gray-50"
        }`}
      >
        <Upload className="w-12 h-12 mx-auto text-indigo-500 mb-3 animate-pulse" />
        <p className="text-gray-600 font-medium">
          Drag & drop your video here, or click to select file.
        </p>
        <p className="text-sm text-gray-400 mt-1">
          MP4 or MOV files under 500MB recommended.
        </p>
      </div>

      <input
        type="file"
        id="video-upload"
        accept="video/*"
        className="hidden"
        onChange={onFileSelected}
      />

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
        <div className="mt-8 space-y-2">
          <p className="text-lg font-medium text-gray-700">
            Uploading & analyzing your video...{" "}
            <span className="font-semibold">{progress}%</span>
          </p>

          {estimatedTime !== null && estimatedTime > 0 && (
            <p className="text-sm text-gray-500">
              ⏳ Estimated time remaining: ~{estimatedTime}s
            </p>
          )}

          <div className="flex items-center justify-between text-xs text-gray-500 mt-1 mb-1">
            <span>Progress</span>
            <span>{progress}%</span>
          </div>

          <div className="w-full bg-gray-200 rounded-full h-3">
            <div
              className="bg-indigo-600 h-3 rounded-full transition-all duration-500 ease-out"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {!!error && <p className="text-red-600 mt-3 font-semibold">{error}</p>}

      {annotatedUrl && (
        <div className="mt-8 space-y-4">
          <p className="text-green-700 font-semibold">
            Analysis complete! Your annotated video is ready.
          </p>

          {/* Full-screen / zoom option via HTML5 video player */}
          <video
            controls
            className="w-full max-h-96 rounded-xl shadow-lg border border-gray-200"
          >
            <source src={annotatedUrl} type="video/mp4" />
            Your browser does not support the video tag.
          </video>
          <p className="text-xs text-gray-500">
            Tip: use the full-screen icon in the player to zoom in while you
            review your delivery.
          </p>

          <div className="flex items-center justify-center gap-3">
            <a
              href={annotatedUrl}
              download="annotated.mp4"
              className="px-5 py-3 bg-gray-900 text-white rounded-xl shadow hover:bg-gray-700"
            >
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

/* ---------- Review Screen (Session feedback) ---------- */

type ReviewScreenProps = {
  onNext: () => void;
  isDarkMode: boolean;
  order: FeedbackKey[];
  onReorder: (order: FeedbackKey[]) => void;
};

const ReviewScreen: React.FC<ReviewScreenProps> = ({
  onNext,
  isDarkMode,
  order,
  onReorder,
}) => {
  const [expanded, setExpanded] = useState<string | null>(null);

  const analysisData: Record<
    FeedbackKey,
    {
      title: string;
      value: string;
      color: TailwindBrand;
      desc: string;
      detail: string;
      icon: LucideIcon;
      timestamp: string;
    }
  > = {
    handMovements: {
      title: "Hand Movements",
      value: "Frequent",
      color: "yellow",
      desc: "Hands are active and sometimes distracting.",
      detail:
        "Try using gestures only when emphasizing key points. Keep hands relaxed near waist level.",
      icon: Hand,
      timestamp: "00:42, 01:18",
    },
    legMovements: {
      title: "Leg Movements",
      value: "Restless",
      color: "red",
      desc: "Notable shifting detected.",
      detail:
        "Frequent weight shifting reduces perceived confidence. Focus on grounding both feet.",
      icon: Move,
      timestamp: "00:57, 01:25",
    },
    eyeGaze: {
      title: "Eye Gaze",
      value: "92%",
      color: "green",
      desc: "Excellent eye contact.",
      detail:
        "Great consistency maintaining connection with the camera across the whole session.",
      icon: Eye,
      timestamp: "No major issues",
    },
    posture: {
      title: "Posture",
      value: "Needs Adjustment",
      color: "red",
      desc: "Slouching detected during certain moments.",
      detail:
        "Posture dipped during moments of hesitation. Keep shoulders open and chin neutral to project confidence.",
      icon: PersonStanding,
      timestamp: "00:33, 01:02",
    },
  };

  const moveUp = (key: FeedbackKey) => {
    const idx = order.indexOf(key);
    if (idx === 0) return;
    const newOrder = [...order];
    [newOrder[idx - 1], newOrder[idx]] = [newOrder[idx], newOrder[idx - 1]];
    onReorder(newOrder);
  };

  return (
    <div className="max-w-5xl mx-auto p-6">
      <h2
        className={`text-3xl font-bold mb-2 ${
          isDarkMode ? "text-slate-50" : "text-gray-900"
        }`}
      >
        3. Session Review
      </h2>
      <p
        className={`font-semibold mb-6 ${
          isDarkMode ? "text-indigo-300" : "text-indigo-600"
        }`}
      >
        AI Analysis Complete! Score: 78/100
      </p>

      {/* WPM explanation */}
      <div
        className={`mb-6 text-sm p-4 rounded-lg flex items-start gap-2 ${
          isDarkMode
            ? "bg-indigo-950/40 text-slate-200"
            : "bg-indigo-50 text-gray-600"
        }`}
      >
        <span className="mt-[2px]">ℹ️</span>
        <p>
          <strong>WPM</strong> stands for <strong>Words Per Minute</strong> and
          measures your speaking pace. A clear, comfortable range for most
          presentations is usually around <strong>110–160 WPM</strong>.
        </p>
      </div>

      <p
        className={`mb-4 text-sm ${
          isDarkMode ? "text-slate-400" : "text-gray-500"
        }`}
      >
        🔁 Use the “Prioritize” arrows to customize the order of your feedback
        sections. Click any card to see detailed insights and timestamps where
        key issues appeared.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {order.map((key) => {
          const metric = analysisData[key];

          return (
            <div
              key={key}
              className={`p-5 rounded-xl border shadow cursor-pointer transition transform hover:-translate-y-[1px] hover:shadow-lg ${
                isDarkMode
                  ? "bg-slate-900 border-slate-800"
                  : "bg-white border-gray-100"
              }`}
              onClick={() => setExpanded(expanded === key ? null : key)}
            >
              <div className="flex justify-between items-center">
                <div className="flex items-center space-x-3">
                  <metric.icon
                    className={`text-${metric.color}-600`}
                    size={22}
                  />
                  <h3
                    className={`text-lg font-semibold ${
                      isDarkMode ? "text-slate-50" : "text-gray-900"
                    }`}
                  >
                    {metric.title}
                  </h3>
                </div>

                <button
                  className="text-indigo-500 text-sm hover:underline"
                  onClick={(e) => {
                    e.stopPropagation();
                    moveUp(key);
                  }}
                >
                  ↑ Prioritize
                </button>
              </div>

              <p
                className={`text-2xl font-bold mt-2 ${
                  isDarkMode ? "text-slate-50" : "text-gray-900"
                }`}
              >
                {metric.value}
              </p>
              <p
                className={`text-sm ${
                  isDarkMode ? "text-slate-400" : "text-gray-500"
                }`}
              >
                {metric.desc}
              </p>

              {expanded === key && (
                <div className="mt-4 bg-gray-50 p-4 rounded-lg text-sm">
                  <p
                    className={
                      isDarkMode ? "text-slate-800" : "text-gray-700"
                    }
                  >
                    <strong>Detailed Insight:</strong> {metric.detail}
                  </p>
                  <p className="mt-2 text-indigo-600">
                    ⏱ Error occurred at: {metric.timestamp}
                  </p>
                  <p
                    className={`mt-1 text-xs ${
                      isDarkMode ? "text-slate-500" : "text-gray-500"
                    }`}
                  >
                    Tip: revisit these timestamps in your annotated video to see
                    exactly what happened.
                  </p>
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div className="mt-10 flex justify-center">
        <button
          onClick={onNext}
          className="px-8 py-3 bg-gray-900 text-white font-semibold rounded-full shadow-lg hover:bg-gray-700 transition transform hover:scale-105"
        >
          View Full Progress Dashboard
        </button>
      </div>
    </div>
  );
};

/* ---------- Analytics / Progress Tracking Screen ---------- */

type AnalyticsScreenProps = {
  onNext: (s: Step) => void;
  isDarkMode: boolean;
  order: FeedbackKey[];
};

const AnalyticsScreen: React.FC<AnalyticsScreenProps> = ({
  onNext,
  isDarkMode,
  order,
}) => {
  type SessionMetric = {
    label: string;
    score: number; // 0–100
    feedback: string;
    emoji: string;
  };

  // histories per feature
  const handHistory: SessionMetric[] = [
    {
      label: "Week 1",
      score: 30,
      feedback:
        "Hands were very active and sometimes covered your face or camera.",
      emoji: "👐",
    },
    {
      label: "Week 2",
      score: 55,
      feedback:
        "Gestures started to match key points, but there’s still extra movement.",
      emoji: "🤏",
    },
    {
      label: "Week 3",
      score: 72,
      feedback:
        "Most gestures felt intentional. Only a few moments of fidgeting.",
      emoji: "👍",
    },
    {
      label: "Week 4",
      score: 85,
      feedback:
        "Great control! Gestures clearly highlighted your main points.",
      emoji: "🌟",
    },
  ];

  const legHistory: SessionMetric[] = [
    {
      label: "Week 1",
      score: 25,
      feedback: "Lots of shifting from foot to foot and visible bouncing.",
      emoji: "🦶",
    },
    {
      label: "Week 2",
      score: 50,
      feedback:
        "Some fidgeting remained, but you stayed planted during key answers.",
      emoji: "🙂",
    },
    {
      label: "Week 3",
      score: 70,
      feedback:
        "Only occasional movement when thinking. Much more stable overall.",
      emoji: "💪",
    },
    {
      label: "Week 4",
      score: 90,
      feedback:
        "Confident, grounded stance throughout almost the entire session.",
      emoji: "🏆",
    },
  ];

  const eyeHistory: SessionMetric[] = [
    {
      label: "Week 1",
      score: 70,
      feedback:
        "Eye contact was good but dropped whenever you glanced at notes.",
      emoji: "👀",
    },
    {
      label: "Week 2",
      score: 78,
      feedback:
        "More consistent eye contact, with fewer long breaks away from camera.",
      emoji: "😊",
    },
    {
      label: "Week 3",
      score: 86,
      feedback:
        "Strong eye contact most of the session, only brief downward glances.",
      emoji: "✨",
    },
    {
      label: "Week 4",
      score: 92,
      feedback:
        "Excellent eye contact across the whole session—felt very engaging.",
      emoji: "💫",
    },
  ];

  const postureHistory: SessionMetric[] = [
    {
      label: "Week 1",
      score: 40,
      feedback:
        "Frequent slouching and leaning forward, especially when you were unsure.",
      emoji: "😬",
    },
    {
      label: "Week 2",
      score: 55,
      feedback:
        "Posture improved during the start, but dipped as the session went on.",
      emoji: "🙂",
    },
    {
      label: "Week 3",
      score: 68,
      feedback:
        "Mostly open posture with only a few moments of rounding the shoulders.",
      emoji: "👍",
    },
    {
      label: "Week 4",
      score: 80,
      feedback:
        "Confident, upright posture through most of the session—big improvement.",
      emoji: "🏅",
    },
  ];

  // which week is selected for each feature
  const [selectedHandIndex, setSelectedHandIndex] = useState(
    handHistory.length - 1
  );
  const [selectedLegIndex, setSelectedLegIndex] = useState(
    legHistory.length - 1
  );
  const [selectedEyeIndex, setSelectedEyeIndex] = useState(
    eyeHistory.length - 1
  );
  const [selectedPostureIndex, setSelectedPostureIndex] = useState(
    postureHistory.length - 1
  );

  const selectedHand = handHistory[selectedHandIndex];
  const selectedLeg = legHistory[selectedLegIndex];
  const selectedEye = eyeHistory[selectedEyeIndex];
  const selectedPosture = postureHistory[selectedPostureIndex];

  const cardShell = (children: React.ReactNode) => (
    <div
      className={`p-6 rounded-xl shadow-lg border ${
        isDarkMode
          ? "bg-slate-900 border-slate-800"
          : "bg-white border-gray-50"
      }`}
    >
      {children}
    </div>
  );

  const renderGraphCard = (key: FeedbackKey) => {
    switch (key) {
      case "handMovements":
        return cardShell(
          <>
            <h3
              className={`text-xl font-semibold mb-2 flex items-center gap-2 ${
                isDarkMode ? "text-slate-50" : "text-gray-800"
              }`}
            >
              <Hand size={20} className="text-indigo-600" />
              <span>Hand Movement Control</span>
              <span className="text-lg">👋</span>
            </h3>
            <p
              className={`text-xs mb-3 ${
                isDarkMode ? "text-slate-400" : "text-gray-500"
              }`}
            >
              Click a bar to see what your AI coach said that week.
            </p>

            <div
              className={`h-44 rounded-lg flex items-end p-3 text-xs relative overflow-hidden ${
                isDarkMode
                  ? "bg-slate-800 text-slate-400"
                  : "bg-gray-100 text-gray-500"
              }`}
            >
              {handHistory.map((session, idx) => {
                const isActive = idx === selectedHandIndex;
                const heightPct = 20 + (session.score / 100) * 70;

                return (
                  <button
                    key={session.label}
                    type="button"
                    onClick={() => setSelectedHandIndex(idx)}
                    className={`flex-1 mx-1 rounded-t-md focus:outline-none transition-transform duration-300 ${
                      isActive
                        ? "bg-indigo-500 scale-105 shadow-lg"
                        : "bg-indigo-200 hover:bg-indigo-300 hover:scale-105"
                    }`}
                    style={{ height: `${heightPct}%` }}
                    title={`Click to view feedback for ${session.label}`}
                  >
                    <span className="sr-only">{session.label}</span>
                  </button>
                );
              })}

              <p className="absolute bottom-2 left-3 text-[10px]">
                Progress over last 4 sessions
              </p>
            </div>

            <div
              className={`mt-4 rounded-lg border p-4 text-sm ${
                isDarkMode
                  ? "bg-indigo-950/40 border-indigo-900 text-slate-100"
                  : "bg-indigo-50 border-indigo-100 text-gray-700"
              }`}
            >
              <p className="font-semibold flex items-center gap-2">
                {selectedHand.emoji} {selectedHand.label} — Score{" "}
                {selectedHand.score}/100
              </p>
              <p className="mt-1">{selectedHand.feedback}</p>
            </div>
          </>
        );

      case "legMovements":
        return cardShell(
          <>
            <h3
              className={`text-xl font-semibold mb-2 flex items-center gap-2 ${
                isDarkMode ? "text-slate-50" : "text-gray-800"
              }`}
            >
              <Move size={20} className="text-indigo-600" />
              <span>Leg Movement Stillness</span>
              <span className="text-lg">🦵</span>
            </h3>
            <p
              className={`text-xs mb-3 ${
                isDarkMode ? "text-slate-400" : "text-gray-500"
              }`}
            >
              Click a bar to see how your stability has changed week to week.
            </p>

            <div
              className={`h-44 rounded-lg flex items-end p-3 text-xs relative overflow-hidden ${
                isDarkMode
                  ? "bg-slate-800 text-slate-400"
                  : "bg-gray-100 text-gray-500"
              }`}
            >
              {legHistory.map((session, idx) => {
                const isActive = idx === selectedLegIndex;
                const heightPct = 20 + (session.score / 100) * 70;

                return (
                  <button
                    key={session.label}
                    type="button"
                    onClick={() => setSelectedLegIndex(idx)}
                    className={`flex-1 mx-1 rounded-t-md focus:outline-none transition-transform duration-300 ${
                      isActive
                        ? "bg-emerald-500 scale-105 shadow-lg"
                        : "bg-emerald-200 hover:bg-emerald-300 hover:scale-105"
                    }`}
                    style={{ height: `${heightPct}%` }}
                    title={`Click to view feedback for ${session.label}`}
                  >
                    <span className="sr-only">{session.label}</span>
                  </button>
                );
              })}

              <p className="absolute bottom-2 left-3 text-[10px]">
                Higher bars = more still and grounded ✅
              </p>
            </div>

            <div
              className={`mt-4 rounded-lg border p-4 text-sm ${
                isDarkMode
                  ? "bg-emerald-950/40 border-emerald-900 text-slate-100"
                  : "bg-emerald-50 border-emerald-100 text-gray-700"
              }`}
            >
              <p className="font-semibold flex items-center gap-2">
                {selectedLeg.emoji} {selectedLeg.label} — Score{" "}
                {selectedLeg.score}/100
              </p>
              <p className="mt-1">{selectedLeg.feedback}</p>
            </div>
          </>
        );

      case "eyeGaze":
        return cardShell(
          <>
            <h3
              className={`text-xl font-semibold mb-2 flex items-center gap-2 ${
                isDarkMode ? "text-slate-50" : "text-gray-800"
              }`}
            >
              <Eye size={20} className="text-indigo-600" />
              <span>Eye Gaze Consistency</span>
              <span className="text-lg">👁️</span>
            </h3>
            <p
              className={`text-xs mb-3 ${
                isDarkMode ? "text-slate-400" : "text-gray-500"
              }`}
            >
              Higher scores mean more consistent, engaging eye contact.
            </p>

            <div
              className={`h-44 rounded-lg flex items-end p-3 text-xs relative overflow-hidden ${
                isDarkMode
                  ? "bg-slate-800 text-slate-400"
                  : "bg-gray-100 text-gray-500"
              }`}
            >
              {eyeHistory.map((session, idx) => {
                const isActive = idx === selectedEyeIndex;
                const heightPct = 20 + (session.score / 100) * 70;

                return (
                  <button
                    key={session.label}
                    type="button"
                    onClick={() => setSelectedEyeIndex(idx)}
                    className={`flex-1 mx-1 rounded-t-md focus:outline-none transition-transform duration-300 ${
                      isActive
                        ? "bg-sky-500 scale-105 shadow-lg"
                        : "bg-sky-200 hover:bg-sky-300 hover:scale-105"
                    }`}
                    style={{ height: `${heightPct}%` }}
                    title={`Click to view feedback for ${session.label}`}
                  >
                    <span className="sr-only">{session.label}</span>
                  </button>
                );
              })}

              <p className="absolute bottom-2 left-3 text-[10px]">
                Higher bars = more consistent eye contact ✅
              </p>
            </div>

            <div
              className={`mt-4 rounded-lg border p-4 text-sm ${
                isDarkMode
                  ? "bg-sky-950/40 border-sky-900 text-slate-100"
                  : "bg-sky-50 border-sky-100 text-gray-700"
              }`}
            >
              <p className="font-semibold flex items-center gap-2">
                {selectedEye.emoji} {selectedEye.label} — Score{" "}
                {selectedEye.score}/100
              </p>
              <p className="mt-1">{selectedEye.feedback}</p>
            </div>
          </>
        );

      case "posture":
      default:
        return cardShell(
          <>
            <h3
              className={`text-xl font-semibold mb-2 flex items-center gap-2 ${
                isDarkMode ? "text-slate-50" : "text-gray-800"
              }`}
            >
              <PersonStanding size={20} className="text-indigo-600" />
              <span>Posture Confidence</span>
              <span className="text-lg">🧍‍♀️</span>
            </h3>
            <p
              className={`text-xs mb-3 ${
                isDarkMode ? "text-slate-400" : "text-gray-500"
              }`}
            >
              These scores capture how open, upright, and confident your posture
              appeared.
            </p>

            <div
              className={`h-44 rounded-lg flex items-end p-3 text-xs relative overflow-hidden ${
                isDarkMode
                  ? "bg-slate-800 text-slate-400"
                  : "bg-gray-100 text-gray-500"
              }`}
            >
              {postureHistory.map((session, idx) => {
                const isActive = idx === selectedPostureIndex;
                const heightPct = 20 + (session.score / 100) * 70;

                return (
                  <button
                    key={session.label}
                    type="button"
                    onClick={() => setSelectedPostureIndex(idx)}
                    className={`flex-1 mx-1 rounded-t-md focus:outline-none transition-transform duration-300 ${
                      isActive
                        ? "bg-rose-500 scale-105 shadow-lg"
                        : "bg-rose-200 hover:bg-rose-300 hover:scale-105"
                    }`}
                    style={{ height: `${heightPct}%` }}
                    title={`Click to view feedback for ${session.label}`}
                  >
                    <span className="sr-only">{session.label}</span>
                  </button>
                );
              })}

              <p className="absolute bottom-2 left-3 text-[10px]">
                Higher bars = more confident, upright posture ✅
              </p>
            </div>

            <div
              className={`mt-4 rounded-lg border p-4 text-sm ${
                isDarkMode
                  ? "bg-rose-950/40 border-rose-900 text-slate-100"
                  : "bg-rose-50 border-rose-100 text-gray-700"
              }`}
            >
              <p className="font-semibold flex items-center gap-2">
                {selectedPosture.emoji} {selectedPosture.label} — Score{" "}
                {selectedPosture.score}/100
              </p>
              <p className="mt-1">{selectedPosture.feedback}</p>
            </div>
          </>
        );
    }
  };

  return (
    <div className="max-w-5xl mx-auto p-6">
      <h2
        className={`text-3xl font-bold mb-2 ${
          isDarkMode ? "text-slate-50" : "text-gray-900"
        }`}
      >
        4. Progress Dashboard
      </h2>
      <p
        className={`mb-6 ${
          isDarkMode ? "text-slate-400" : "text-gray-600"
        }`}
      >
        These graphs mirror your Review feedback and show how each area has
        changed over time. The order matches your prioritized feedback.
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {order.map((key) => (
          <React.Fragment key={key}>{renderGraphCard(key)}</React.Fragment>
        ))}
      </div>

      <div className="mt-10 text-center">
        <p
          className={`text-lg font-medium mb-2 ${
            isDarkMode ? "text-slate-100" : "text-gray-700"
          }`}
        >
          🎯 Ready for your next practice?
        </p>
        <button
          onClick={() => onNext(2)}
          className="inline-flex items-center gap-2 px-8 py-3 bg-indigo-600 text-white font-semibold rounded-full shadow-lg hover:bg-indigo-700 transition-transform duration-200 hover:scale-105"
        >
          Start New Practice
          <ArrowRight size={18} />
        </button>
      </div>
    </div>
  );
};

/* ---------- Main App ---------- */

const App: React.FC = () => {
  const [step, setStep] = useState<Step>(1);
  const [isDarkMode, setIsDarkMode] = useState(false);

  // shared priority order between Review + Analytics
  const [feedbackOrder, setFeedbackOrder] = useState<FeedbackKey[]>([
    "handMovements",
    "legMovements",
    "eyeGaze",
    "posture",
  ]);

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
        return (
          <ReviewScreen
            onNext={() => navigateTo(5)}
            isDarkMode={isDarkMode}
            order={feedbackOrder}
            onReorder={setFeedbackOrder}
          />
        );
      case 5:
        return (
          <AnalyticsScreen
            onNext={navigateTo}
            isDarkMode={isDarkMode}
            order={feedbackOrder}
          />
        );
      default:
        return <WelcomeScreen onStart={() => navigateTo(2)} />;
    }
  };

  return (
    <div
      className={`min-h-dvh font-sans antialiased flex flex-col transition-colors duration-300 ${
        isDarkMode ? "bg-slate-950 text-slate-50" : "bg-gray-50 text-gray-900"
      }`}
    >
      <Header
        onNavigate={navigateTo}
        currentStep={step}
        isDarkMode={isDarkMode}
        onToggleTheme={() => setIsDarkMode((d) => !d)}
      />

      {/* Fill space between sticky header and bottom nav */}
      <main className="flex-1 py-10 pb-24">
        <div className="mx-auto px-4 sm:px-6 lg:px-8 h-full">
          <div
            className="bg-white p-8 sm:p-12 rounded-2xl shadow-2xl
                       min-h-[calc(100dvh-128px)] flex items-center justify-center"
          >
            {renderContent()}
          </div>
        </div>
      </main>

      <BottomNav currentStep={step} onNavigate={navigateTo} />
    </div>
  );
};

export default App;
