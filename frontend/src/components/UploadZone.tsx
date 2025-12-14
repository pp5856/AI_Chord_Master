import React, { useState } from 'react';
import { Upload, Youtube, Music, Loader2 } from 'lucide-react';

interface UploadZoneProps {
  onUpload: (file: File) => void;
  onYoutube: (url: string) => void;
  isLoading: boolean;
}

export const UploadZone: React.FC<UploadZoneProps> = ({ onUpload, onYoutube, isLoading }) => {
  const [activeTab, setActiveTab] = useState<'file' | 'youtube'>('file');
  const [dragActive, setDragActive] = useState(false);
  const [youtubeUrl, setYoutubeUrl] = useState('');

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onUpload(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      onUpload(e.target.files[0]);
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
      {/* Tabs */}
      <div className="flex border-b border-gray-200">
        <button
          className={`flex-1 py-4 text-sm font-medium flex items-center justify-center gap-2 transition-colors ${
            activeTab === 'file' ? 'text-green-600 border-b-2 border-green-600 bg-green-50/50' : 'text-gray-500 hover:text-gray-700'
          }`}
          onClick={() => setActiveTab('file')}
        >
          <Music size={18} />
          오디오 파일
        </button>
        <button
          className={`flex-1 py-4 text-sm font-medium flex items-center justify-center gap-2 transition-colors ${
            activeTab === 'youtube' ? 'text-red-600 border-b-2 border-red-600 bg-red-50/50' : 'text-gray-500 hover:text-gray-700'
          }`}
          onClick={() => setActiveTab('youtube')}
        >
          <Youtube size={18} />
          유튜브 링크
        </button>
      </div>

      {/* Content */}
      <div className="p-8">
        {isLoading ? (
          <div className="flex flex-col items-center justify-center py-12 text-gray-500">
            <Loader2 size={48} className="animate-spin text-green-500 mb-4" />
            <p className="text-lg font-medium">음악을 분석하고 있습니다...</p>
            <p className="text-sm opacity-70">잠시만 기다려주세요 (약 10~30초)</p>
          </div>
        ) : activeTab === 'file' ? (
          <div
            className={`relative border-2 border-dashed rounded-lg p-12 text-center transition-all ${
              dragActive ? 'border-green-500 bg-green-50' : 'border-gray-300 hover:border-gray-400 hover:bg-gray-50'
            }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <input
              type="file"
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              onChange={handleChange}
              accept="audio/*"
            />
            <div className="flex flex-col items-center gap-4 pointer-events-none">
              <div className="w-16 h-16 bg-green-100 text-green-600 rounded-full flex items-center justify-center">
                <Upload size={32} />
              </div>
              <div>
                <p className="text-lg font-medium text-gray-900">파일을 드래그하거나 클릭하여 업로드</p>
                <p className="text-sm text-gray-500 mt-1">MP3, WAV 파일 지원</p>
              </div>
            </div>
          </div>
        ) : (
          <div className="py-8">
            <div className="flex gap-2">
              <input
                type="text"
                placeholder="https://www.youtube.com/watch?v=..."
                className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-transparent outline-none transition-all"
                value={youtubeUrl}
                onChange={(e) => setYoutubeUrl(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && onYoutube(youtubeUrl)}
              />
              <button
                onClick={() => onYoutube(youtubeUrl)}
                className="px-6 py-3 bg-red-600 text-white font-medium rounded-lg hover:bg-red-700 transition-colors flex items-center gap-2"
              >
                분석하기
              </button>
            </div>
            <p className="text-sm text-gray-500 mt-3 ml-1">
              * 유튜브 영상의 오디오를 추출하여 분석합니다.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
