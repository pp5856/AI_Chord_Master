import React, { useEffect, useRef } from 'react';

interface ChordEvent {
  start: number;
  end: number;
  chord: string;
}

interface SheetMusicDisplayProps {
  chords: ChordEvent[];
  currentTime: number;
  onSeek: (time: number) => void;
}

export const SheetMusicDisplay: React.FC<SheetMusicDisplayProps> = ({ chords, currentTime, onSeek }) => {
  const scrollRef = useRef<HTMLDivElement>(null);

  // 현재 재생 중인 코드 찾기
  const currentChordIndex = chords.findIndex(
    c => currentTime >= c.start && currentTime < c.end
  );

  // 자동 스크롤
  useEffect(() => {
    if (currentChordIndex !== -1 && scrollRef.current) {
      const activeElement = scrollRef.current.children[currentChordIndex] as HTMLElement;
      if (activeElement) {
        activeElement.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'center' });
      }
    }
  }, [currentChordIndex]);

  return (
    <div className="w-full bg-white rounded-xl shadow-sm border border-gray-200 p-8 overflow-hidden">
      <h3 className="text-lg font-bold text-gray-800 mb-6 flex items-center gap-2">
        🎼 코드 악보
      </h3>
      
      <div 
        ref={scrollRef}
        className="flex flex-wrap gap-y-12 gap-x-0 overflow-y-auto max-h-[400px] p-4"
      >
        {chords.map((chord, index) => {
          const isActive = index === currentChordIndex;
          
          return (
            <div 
              key={index}
              onClick={() => onSeek(chord.start)}
              className={`
                relative w-32 h-24 border-b-2 border-gray-800 flex items-end pb-2 justify-center
                cursor-pointer transition-all duration-200 group
                ${isActive ? 'bg-green-50' : 'hover:bg-gray-50'}
              `}
            >
              {/* 마디 구분선 (오른쪽) */}
              <div className="absolute right-0 bottom-0 w-[2px] h-full bg-gray-800 opacity-20 group-hover:opacity-40"></div>
              
              {/* 코드 이름 */}
              <div className={`
                text-2xl font-bold mb-4
                ${isActive ? 'text-green-600 scale-110' : 'text-gray-800'}
              `}>
                {chord.chord.replace('_', '')}
              </div>

              {/* 시간 표시 (호버 시) */}
              <div className="absolute top-2 left-2 text-xs text-gray-400 opacity-0 group-hover:opacity-100">
                {Math.floor(chord.start / 60)}:{Math.floor(chord.start % 60).toString().padStart(2, '0')}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
