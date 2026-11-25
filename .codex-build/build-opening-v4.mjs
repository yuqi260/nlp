import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';

const skillDir='C:/Users/USER/.codex/plugins/cache/openai-primary-runtime/presentations/26.1111.10930/skills/presentations';
const workspaceDir='C:/Users/USER/Desktop/nlp';
const finalPath=path.join(workspaceDir,'作业二','狼人杀身份推断_开题汇报_v5.pptx');
const blue='#00539B', red='#C00000', dark='#111111', muted='#555555';
const font='微软雅黑';
const p=await PresentationFile.importPptx(await FileBlob.load('C:/Users/USER/Desktop/26C02_概率与概率图模型.pptx'));

function addText(slide,text,x,y,w,h,size=26,color=dark,bold=false,align='left'){
  const s=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{style:'solid',fill:'none',width:0}});
  s.text=text;
  s.text.style={typeface:font,fontSize:size,color,bold,alignment:align,verticalAlignment:'middle',wrap:'square',autoFit:'shrinkText',insets:0};
  return s;
}
function addBody(slide,title,num){
  addText(slide,title,58,41,995,60,37,blue,true);
  addText(slide,String(num),1052,680,24,22,18,dark,false,'right');
}
function line(slide,label,detail,y,{accent=blue,size=27}={}){
  addText(slide,label,65,y,235,43,size,accent,true);
  addText(slide,detail,310,y,745,62,25,dark,false);
}
async function addPhoto(slide,file,x,y,w,h,alt){
  const b=await fs.readFile(path.join(workspaceDir,'作业二','assets',file));
  slide.images.add({blob:b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength),contentType:'image/jpeg',alt,fit:'cover',position:{left:x,top:y,width:w,height:h}});
}

// Keep the imported artwork and layout; replace only the lecture-specific text.
const cover=p.slides.items[0];
addText(cover,'狼人杀玩家身份推断系统',64,323,990,78,51,'#FFFFFF',true,'center');
addText(cover,'融合多轮对话与行为信息',110,415,902,46,32,'#FFFFFF',false,'center');
addText(cover,'NLP 课程大作业 · 开题汇报',110,493,902,36,23,'#FFFFFF',false,'center');
cover.speakerNotes.text='开题重点：说明题目合理性、问题边界、方法、可行性与预期效果。';

const toc=p.slides.items[1];
p.delete(['sh/xcvil0j2','im/98r25g7i','im/n69k36ps','im/lk721w7m','im/zipkzm9g','im/dg72xwra','im/2lczyxon','im/oneh0n6t']);
const tocRows=['选题背景与任务定义','相关研究与项目定位','核心思想与技术路径','数据来源与可行性','实验设计与预期效果','实施安排'];
tocRows.forEach((t,i)=>{
  addText(toc,'□',524,239+i*60,35,37,29,red,true);
  addText(toc,t,568,238+i*60,495,43,26,i===0?red:dark,true);
});
toc.speakerNotes.text='先说明任务背景与挑战，再讲方案和验证办法。';

const base=p.slides.items[2];
p.delete(['sh/e10f2twf','sh/l4fi9cza','sh/x0vi5czy']);
const blank=base.copy();
for(let i=0;i<8;i++) p.slides.addCopy(blank);
const slides=p.slides.items.slice(2);

// 3. Background
{
 const s=slides[0]; addBody(s,'选题背景：语言、行为与策略交织',3);
 addText(s,'多轮对话',66,172,203,45,26,blue,true);addText(s,'发言和立场会随着游戏推进而变化。',270,170,493,65,25);
 addText(s,'行为信息',66,278,203,45,26,blue,true);addText(s,'投票和公开结果可用来核对发言。',270,276,493,65,25);
 addText(s,'策略性表达',66,384,203,45,26,blue,true);addText(s,'玩家可能隐瞒或误导，叙述自洽也未必真实。',270,375,493,79,25);
 await addPhoto(s,'students_board_game.jpg',792,176,265,264,'真实桌游讨论照片，仅作情境示意');
 addText(s,'桌游讨论场景（示意）',805,447,244,29,17,muted);
 addText(s,'例如：一名玩家口头怀疑 7 号，投票却给了 5 号；下一轮又改变看法。\n判断其身份，需要把前后发言、投票和其他线索放在一起。',69,527,982,110,24,red,true);
 s.speakerNotes.text='照片：Evan Marvell / Unsplash，https://unsplash.com/photos/people-playing-a-board-game-at-a-table-indoors-aHketkfrSFg 。照片并非狼人杀对局。例子为虚构，仅用于说明语言与行为可能不一致；不能仅凭这一处矛盾定身份。';
}
// 4. Task definition
{
 const s=slides[1]; addBody(s,'任务定义：根据公开信息推断玩家身份',4);
 line(s,'观察时点','例如第二天投票结束后，判断当时每个人的身份。',172);
 line(s,'系统输入','截至该时点的公开发言、投票及游戏结果。',278);
 line(s,'系统输出','每位玩家的具体身份判断，并附上相应依据。',384);
 addText(s,'只使用当时所有人都能知道的信息；真实身份仅用于事后检验。',70,541,968,95,27,red,true);
 s.speakerNotes.text='优先选规则一致的九人游戏数据；具体人数和角色以数据核验为准。系统也可以保留备选身份。';
}
// 5. Related work
{
 const s=slides[2]; addBody(s,'相关研究：已有成果与本项目切入点',5);
 addText(s,'研究方向',65,157,246,43,25,blue,true);
 addText(s,'已有成果',324,157,315,43,25,blue,true);
 addText(s,'本项目还需完成',650,157,405,43,25,blue,true);
 const rows=[
  ['早期身份识别','根据语言识别隐藏的欺骗者','进一步区分每人的具体身份'],
  ['阿瓦隆长对话研究','结合多轮对话与游戏进展','适配狼人杀的规则和角色'],
  ['MaKTO 狼人杀智能体','训练模型参与游戏的能力','从旁观者角度逐轮判断身份'],
  ['CSP4SDG 身份推断','结合规则与发言线索推断身份','核验中文对局及公开信息处理']
 ];
 rows.forEach((r,i)=>{const y=223+i*91; addText(s,r[0],65,y,245,70,22,dark,true);addText(s,r[1],324,y,315,70,22,dark);addText(s,r[2],650,y,405,70,22,red);});
 addText(s,'本项目：整理公开记录，综合多轮线索，预测具体身份，并检查判断依据。',65,626,980,45,23,blue,true);
 s.speakerNotes.text='来源：Girlea et al. 2016 https://aclanthology.org/N16-1047/ ; Ibraheem et al. 2022 https://aclanthology.org/2022.naacl-main.11/ ; Stepputtis et al. 2023 https://aclanthology.org/2023.findings-emnlp.748/ ; MaKTO https://arxiv.org/abs/2501.14225 ; CSP4SDG https://arxiv.org/abs/2511.06175 。此处描述任务范围差异，不宣称前人方法存在未经证实的性能缺陷。';
}
// 6. Core approach
{
 const s=slides[3]; addBody(s,'核心思想：把多轮发言与行为联系起来',6);
 line(s,'发言记录','整理谁声明了什么、怀疑谁、看法何时发生变化。',174);
 line(s,'投票与事件','将实际行为与发言对照，并按时间顺序分析。',279);
 line(s,'角色人数','判断结果要符合游戏规定的各类身份人数。',384);
 line(s,'依据回查','每个判断都能对应到原始发言或公开事件。',489);
 s.speakerNotes.text='身份声明不等于事实。投票不能单独决定阵营。全局角色配额用于输出检查；是否提高准确率需要实验验证。';
}
// 7. Technical route
{
 const s=slides[4]; addBody(s,'技术路径：从原始对局到逐轮预测',7);
 const steps=[['① 数据处理','筛选公开信息，统一玩家编号与事件顺序'],['② 按轮次截取','只保留当前判断时点之前的记录'],['③ 整理历史','保留近期原文，概括各玩家此前的发言与行为'],['④ 模型推断','用中文大语言模型判断所有玩家的身份'],['⑤ 检查展示','核对角色人数和证据，展示每轮判断变化']];
 steps.forEach((r,i)=>{const y=164+i*96;addText(s,r[0],67,y,250,50,27,blue,true);addText(s,r[1],324,y,730,61,25,dark);});
 addText(s,'先测试模型直接判断，再检验历史信息整理的作用；模型微调视条件决定。',70,635,990,49,23,red,true);
 s.speakerNotes.text='技术路径为计划，当前尚未实现系统或跑出结果。普通模型API可用于基线推断；LoRA训练需要单独的训练环境。';
}
// 8. Data
{
 const s=slides[5]; addBody(s,'数据方案：先核验，再构建公开输入',8);
 addText(s,'候选来源：MaKTO-Werewolf 公开对局数据',68,165,980,50,29,blue,true);
 addText(s,'允许进入模型',70,263,440,50,26,blue,true);
 addText(s,'必须隔离',571,263,455,50,26,red,true);
 addText(s,'公开发言\n公开投票与出局\n主持人已发布的公告',70,324,445,215,26,dark);
 addText(s,'玩家真实身份和私下得知的信息\n尚未公开的夜间行动\n判断时点之后的事件',571,324,455,215,26,dark);
 addText(s,'先统计规则与有效对局，再人工抽查公开信息提取结果。',70,581,980,63,25,red,true);
 s.speakerNotes.text='数据来源：https://huggingface.co/datasets/ReneeYe/werewolf_game_reasoning 。部分公开样例包含私人信息，因此不能直接把现成prompt当旁观者输入。尚未下载、统计有效样本或核验最终许可。';
}
// 9. Evaluation
{
 const s=slides[6]; addBody(s,'实验设计：比较方法是否真的有帮助',9);
 line(s,'随机猜测','按每种身份的人数随机分配。',168);
 line(s,'直接判断','模型直接读取截至当前时点的公开记录。',264);
 line(s,'整理后判断','先整理每个人的历史发言和行为，再交给模型。',360);
 addText(s,'评价：每种身份识别得怎样、是否整局全对、不同轮次表现如何。',69,502,970,85,26,blue,true);
 addText(s,'按整局划分数据；所有方案使用相同对局和预测时点。',69,596,970,50,25,red,true);
 s.speakerNotes.text='结果尚未产生，不能写预期准确率。可增加合法输出率与证据引用有效率。LoRA及投票消融视条件补充。';
}
// 10. Expected output
{
 const s=slides[7]; addBody(s,'预期效果：可查看身份、证据与变化',10);
 addText(s,'预期系统界面（示意）',71,156,860,44,23,red,true);
 addText(s,'选择对局与轮次',78,237,310,52,28,blue,true);
 addText(s,'查看该时点的公开记录',399,237,610,52,26,dark);
 addText(s,'逐玩家身份表',78,340,310,52,28,blue,true);
 addText(s,'预测身份、候选身份、证据引用',399,340,610,52,26,dark);
 addText(s,'轮次变化',78,443,310,52,28,blue,true);
 addText(s,'比较上一轮判断，回查原始事件',399,443,610,52,26,dark);
 addText(s,'同时提交可复现的对照实验、分角色结果和典型错误案例。',72,586,960,67,26,red,true);
 s.speakerNotes.text='预期效果是系统能力和交付物，非已完成成果。早期公开信息可能不足以唯一识别平民与隐藏神职，应允许候选身份。';
}
// 11. Plan
{
 const s=slides[8]; addBody(s,'实施安排与可行性',11);
 line(s,'第一阶段','核验数据与规则，按轮次整理公开记录。',170);
 line(s,'第二阶段','建立模型基线，加入历史信息组织和结果检查。',272);
 line(s,'第三阶段','完成原型、对照实验与错误分析；条件允许时微调。',374);
 addText(s,'主要风险',67,500,221,50,27,red,true);
 addText(s,'混入不公开信息 → 筛选数据并人工抽查\n有效样本不足 → 选择记录完整的一种规则\n部分身份难以确定 → 保留候选并如实评测',300,494,748,155,24,dark);
 s.speakerNotes.text='当前仍处于开题阶段。可行性来自明确的输入输出、候选数据和模型路线；有效样本与最终效果需后续验证。';
}

await fs.mkdir(path.join(workspaceDir,'.codex-finalizer'),{recursive:true});
await fs.mkdir(path.dirname(finalPath),{recursive:true});
const candidatePath=path.join(workspaceDir,'.codex-finalizer','opening-v5-candidate.pptx');
await (await PresentationFile.exportPptx(p)).save(candidatePath);
const {finalizePresentation}=await import(pathToFileURL(path.join(skillDir,'container_tools/artifact_tool_utils.mjs')).href);
const result=await finalizePresentation({workspaceDir,candidatePath,finalPath,pythonExecutable:'C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1111-570-0_26.1111.570.0_x64__3k8sg7r9htsxt/dependencies/python/python.exe',integrityValidatorPath:path.join(skillDir,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skillDir,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','10691495,7559675','--validate-heading-fit'],sourceTemplatePath:'C:/Users/USER/Desktop/26C02_概率与概率图模型.pptx',verifyArtifactToolImport:true,receiptPath:path.join(workspaceDir,'.codex-finalizer','opening-v5-validation.json')});
console.log(JSON.stringify({slideCount:p.slides.items.length,finalPath,result},null,2));
