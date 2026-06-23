import re
order=["01_introduction","02_related_work","03_system_model","04_framework","05_theory","06_simulation","07_conclusion"]
def s1(s):
    l=s.splitlines(); return "\n".join(l[1:]) if l and l[0].lstrip().startswith("%") and "notation" not in l[0] else s
macros=open("_section_sources/notation_macros.tex").read()
bodies=[s1(open(f"_section_sources/{f}.tex").read()) for f in order]
appendix=s1(open("_section_sources/08_appendix.tex").read())
full=open("la_jssa.tex").read()
header=full.split("%================= NOTATION MACROS")[0]+"%================= NOTATION MACROS (single source of truth) =====\n"
m=re.search(r"(\\begin\{document\}.*?)(?=\n\\section|\\IEEEPARstart)", full, re.S)
out=header+macros+"\n"+m.group(1)+"\n"+"\n\n".join(bodies)+"\n\\appendices\n"+appendix+"\n{\\small\n\\bibliographystyle{IEEEtran}\n\\bibliography{references}\n}\n\n\\end{document}\n"
open("la_jssa.tex","w").write(out); print("rebuilt (small bib preserved)")
