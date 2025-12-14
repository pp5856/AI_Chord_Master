import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { UploadZone } from './components/UploadZone';
import { SheetMusicDisplay } from './components/SheetMusicDisplay';
import { Music, ArrowLeft } from 'lucide-react';

interface ChordResult {
  start: number;
  end: number;
  chord: string;
}

interface AnalysisResponse {
  success: boolean;
  tempo: number;
  results: ChordResult[];
  error?: string;
}

function App() {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [audioSrc, setAudioSrc] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  
  const audioRef = useRef<HTMLAudioElement>(null);

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
    }
  };

  const handleSeek = (time: number) => {
    if (audioRef.current) {
      audioRef.current.currentTime = time;
      audioRef.current.play();
    }
  };

  const handleUpload = async (file: File) => {
    setIsLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      // 1. 오디오 미리보기 설정
      const objectUrl = URL.createObjectURL(file);
      setAudioSrc(objectUrl);

      // 2. 서버 분석 요청
      const response = await axios.post('http://localhost:5000/analyze/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      setResult(response.data);
    } catch (error) {
      console.error(error);
      alert('분석 중 오류가 발생했습니다.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleYoutube = async (url: string) => {
    setIsLoading(true);
    try {
      const response = await axios.post('http://localhost:5000/analyze/youtube', { url });
      
      // 유튜브는 서버에서 다운로드된 파일 경로를 반환해야 하지만, 
      // 현재 구조상 로컬 파일 재생이 어려우므로 (브라우저 보안),
      // 실제 서비스에서는 스트리밍 URL을 반환해야 합니다.
      // 여기서는 분석 결과만 보여주는 것으로 가정합니다.
      setResult(response.data);
      alert('유튜브 분석 완료! (오디오 재생은 현재 로컬 파일만 지원됩니다)');
    } catch (error) {
      console.error(error);
      alert('유튜브 분석 실패');
    } finally {
      setIsLoading(false);
    }
  };

  const reset = () => {
    setResult(null);
    setAudioSrc(null);
    setCurrentTime(0);
  };

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900 font-sans">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-5xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2 cursor-pointer" onClick={reset}>
            <div className="w-8 h-8 bg-green-600 rounded-lg flex items-center justify-center text-white">
              <Music size={20} />
            </div>
            <h1 className="text-xl font-bold tracking-tight">AI Band Master</h1>
          </div>
          {result && (
            <button 
              onClick={reset}
              className="text-sm text-gray-500 hover:text-gray-900 flex items-center gap-1"
            >
              <ArrowLeft size={16} />
              처음으로
            </button>
          )}
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-5xl mx-auto px-6 py-12">
        {!result ? (
          <div className="flex flex-col items-center justify-center min-h-[60vh]">
            <h2 className="text-3xl font-bold text-gray-900 mb-4 text-center">
              당신의 음악을 <span className="text-green-600">AI 악보</span>로 변환하세요
            </h2>
            <p className="text-gray-500 mb-12 text-center max-w-lg">
              MP3 파일을 업로드하거나 유튜브 링크를 입력하면,<br/>
              인공지능이 코드를 분석하여 실시간 악보를 만들어드립니다.
            </p>
            
            <UploadZone 
              onUpload={handleUpload} 
              onYoutube={handleYoutube}
              isLoading={isLoading}
            />
          </div>
        ) : (
          <div className="space-y-8 animate-fade-in">
            {/* Audio Player */}
            {audioSrc && (
              <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200 sticky top-20 z-20">
                <audio 
                  ref={audioRef}
                  src={audioSrc} 
                  controls 
                  className="w-full"
                  onTimeUpdate={handleTimeUpdate}
                />
              </div>
            )}

            {/* Sheet Music */}
            <SheetMusicDisplay 
              chords={result.results} 
              currentTime={currentTime}
              onSeek={handleSeek}
            />
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
