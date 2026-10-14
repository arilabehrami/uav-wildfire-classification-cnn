// Decorative orbital illustration; it does not represent live monitoring data.
export default function OrbitalScene() {
  return (
    <div className="orbital-scene" aria-hidden="true">
      <svg viewBox="0 0 600 380" fill="none">
        <defs>
          <radialGradient id="earth-ocean" cx="28%" cy="25%" r="80%">
            <stop stopColor="#214f7b" />
            <stop offset=".48" stopColor="#102b4a" />
            <stop offset="1" stopColor="#030911" />
          </radialGradient>
          <radialGradient id="earth-shade" cx="23%" cy="25%" r="82%">
            <stop offset=".25" stopColor="#020710" stopOpacity="0" />
            <stop offset=".8" stopColor="#020710" stopOpacity=".48" />
            <stop offset="1" stopColor="#020710" stopOpacity=".95" />
          </radialGradient>
          <radialGradient id="earth-halo">
            <stop offset=".65" stopColor="#297bed" stopOpacity=".12" />
            <stop offset=".78" stopColor="#297bed" stopOpacity=".08" />
            <stop offset="1" stopColor="#297bed" stopOpacity="0" />
          </radialGradient>
          <clipPath id="earth-disc"><circle cx="315" cy="190" r="133" /></clipPath>
          <pattern id="earth-dots" width="6" height="6" patternUnits="userSpaceOnUse">
            <circle cx="2" cy="2" r=".7" fill="#95cced" fillOpacity=".4" />
          </pattern>
        </defs>
        <g fill="#9bb8d7">
          {[[78,65],[153,109],[476,43],[523,147],[490,290],[124,283],[384,27],[64,205],[550,333],[227,32]].map(([x,y], i) => (
            <circle key={i} cx={x} cy={y} r={i % 3 === 0 ? 1.4 : .8} opacity=".5" />
          ))}
        </g>
        <circle cx="315" cy="190" r="185" fill="url(#earth-halo)" />
        <g stroke="#456d9a" strokeWidth=".7">
          <ellipse cx="315" cy="190" rx="231" ry="99" transform="rotate(-24 315 190)" strokeDasharray="3 7" opacity=".5" />
          <ellipse cx="315" cy="190" rx="191" ry="158" transform="rotate(-24 315 190)" opacity=".2" />
          <path d="M315 13v22M315 345v22M63 190h22M545 190h22" opacity=".45" />
        </g>
        <circle cx="315" cy="190" r="134" stroke="#74b7f3" strokeOpacity=".45" />
        <circle cx="315" cy="190" r="133" fill="url(#earth-ocean)" />
        <g clipPath="url(#earth-disc)">
          <g fill="#497a91" stroke="#71a4bb" strokeWidth=".7" strokeLinejoin="round">
            <path d="M269 60l-22 17-8 18 13 11 14-7 9-20 17-7 9-15zM281 101l-17 11-11 21 9 14 14-8 8 8-1 12 23 6 15-6 12 8 12-10-14-16-17-3-8-16 13-14-7-16-14 10-5 16z" />
            <path d="M273 162l-23 4-19 19 1 23 18 18 24 3 14 17 9 32 15 22 14-6 17-30 6-28 16-18-10-19-20-8-13-21-26-9zM354 252l-5 20 4 15 9-15 1-16z" />
            <path d="M323 66l16-17 37 3 42 23 22 23 18 46-17 20-28-4-13 12-25-12-18 6-13-18-9-23-17-12 12-15zM359 175l22-3 18 16 7 28-12 8-15-23-9-6zM410 218l11 6 11 24-8 8-12-17z" />
            <path d="M181 99l25-10 20 15-4 21-19 5-9 18-18-1zM187 172l23 8 14 28-7 28-16 27-8 27-14-15-7-43z" />
          </g>
          <circle cx="315" cy="190" r="133" fill="url(#earth-dots)" />
          <g stroke="#80b5d9" strokeWidth=".6" opacity=".19">
            <ellipse cx="315" cy="190" rx="52" ry="133" />
            <ellipse cx="315" cy="190" rx="104" ry="133" />
            <ellipse cx="315" cy="190" rx="133" ry="46" />
            <ellipse cx="315" cy="190" rx="133" ry="94" />
            <path d="M182 190h266M315 57v266" />
          </g>
          <circle cx="315" cy="190" r="133" fill="url(#earth-shade)" />
        </g>
        <path d="M105 258C133 310 281 293 420 226S548 104 524 102" stroke="#659ce0" strokeOpacity=".65" />
        <g className="orbital-beacon">
          <circle cx="299" cy="143" r="15" stroke="#7cbbff" strokeOpacity=".25" />
          <circle cx="299" cy="143" r="7" stroke="#a5d3ff" strokeOpacity=".7" />
          <circle cx="299" cy="143" r="2.5" fill="#d7edff" />
        </g>
        <path d="M299 143l54-50h70" stroke="#98c6f6" strokeWidth=".7" strokeOpacity=".55" />
        <g transform="translate(465 181) rotate(-28)">
          <path d="M-21 0h42M0-20v40" stroke="#bcd8f5" strokeWidth="3" />
          <rect x="-7" y="-9" width="14" height="18" rx="3" fill="#d4e7f8" />
          <path d="M-33-10h18v20h-18zM15-10h18v20H15z" fill="#173b66" stroke="#78aee8" />
          <path d="M-27-10v20M-21-10v20M21-10v20M27-10v20M-33 0h18M15 0h18" stroke="#78aee8" strokeWidth=".5" />
        </g>
      </svg>
    </div>
  );
}
