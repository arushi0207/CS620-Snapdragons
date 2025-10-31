import React from "react";
import type { LucideIcon } from "lucide-react";
import { Home, Target, Upload, Smile, TrendingUp } from "lucide-react";

type Step = 1 | 2 | 3 | 4 | 5;

type Item = {
  step: Step;
  label: string;
  icon: LucideIcon;
};

const items: Item[] = [
  { step: 1, label: "Home",      icon: Home },
  { step: 2, label: "Goals",     icon: Target },
  { step: 3, label: "Upload",    icon: Upload },
  { step: 4, label: "Review",    icon: Smile },
  { step: 5, label: "Analytics", icon: TrendingUp },
];

export default function BottomNav({
  currentStep,
  onNavigate,
}: {
  currentStep: Step;
  onNavigate: (s: Step) => void;
}) {
  return (
    <nav
      className="
        fixed bottom-0 left-0 right-0 z-20 sm:hidden
        bg-white/90 backdrop-blur border-t border-gray-200 shadow-lg
        pb-[env(safe-area-inset-bottom)]  /* safe area on iOS */
      "
    >
      <ul className="grid grid-cols-5 max-w-md mx-auto">
        {items.map(({ step, label, icon: Icon }) => {
          const active = currentStep === step;
          return (
            <li key={label}>
              <button
                onClick={() => onNavigate(step)}
                className={`
                  w-full flex flex-col items-center justify-center py-2.5
                  text-xs font-medium
                  ${active ? "text-indigo-600" : "text-gray-600 hover:text-indigo-600"}
                `}
              >
                <Icon size={22} className={active ? "stroke-indigo-600" : "stroke-current"} />
                <span className="mt-1">{label}</span>
              </button>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}