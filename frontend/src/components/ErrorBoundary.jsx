import { Component } from "react";

// Pa kete, nje gabim render-i brenda hartes se nxehtesise e zbardh gjithe
// faqen pa shpjegim. Kufiri e mban aplikacionin te lexueshem dhe e tregon
// arsyen, pa pretenduar qe rezultati i fundit mbetet i vlefshem.
class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { failure: null };
  }

  static getDerivedStateFromError(failure) {
    return { failure };
  }

  componentDidCatch(failure, info) {
    console.error("Gabim render-i:", failure, info?.componentStack);
  }

  render() {
    if (!this.state.failure) return this.props.children;

    return (
      <div className="boundary">
        <h1>Ndërfaqja ndaloi</h1>
        <p>
          Ndodhi një gabim gjatë vizatimit të faqes. Klasifikimet e kryera
          ruhen në regjistër; rifresko faqen për të vazhduar.
        </p>
        <pre>{String(this.state.failure?.message ?? this.state.failure)}</pre>
        <button
          type="button"
          className="btn primary"
          onClick={() => window.location.reload()}
        >
          Rifresko faqen
        </button>
      </div>
    );
  }
}

export default ErrorBoundary;
