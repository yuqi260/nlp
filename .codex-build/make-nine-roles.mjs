import fs from 'node:fs/promises';
import path from 'node:path';
import sharp from 'sharp';

const root='C:/Users/USER/Desktop/nlp/作业二';
const iconRoot=path.join(root,'assets','role-icons');
const out=path.join(root,'九人狼人杀身份配置');
const blue='#07549C',red='#A33742',ink='#192B42',grey='#526176';
const esc=s=>s.replaceAll('&','&amp;').replaceAll('<','&lt;');
const text=(s,x,y,size=24,color=ink,weight=400,anchor='start')=>`<text x="${x}" y="${y}" fill="${color}" font-size="${size}" font-weight="${weight}" text-anchor="${anchor}">${esc(s)}</text>`;
async function icon(name,x,y,size,color){
 let raw=await fs.readFile(path.join(iconRoot,name+'.svg'),'utf8');
 let content=raw.replace(/^[\s\S]*?<svg[^>]*>/,'').replace(/<\/svg>\s*$/,'');
 if(name==='wolf-head'){
  content=content.replace(/<path d="M0 0h512v512H0z"\s*\/>/,'').replaceAll('fill="#fff"',`fill="${color}"`);
  return `<svg x="${x}" y="${y}" width="${size}" height="${size}" viewBox="0 0 512 512">${content}</svg>`;
 }
 return `<svg x="${x}" y="${y}" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="${color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">${content}</svg>`;
}

let b=[];
b.push(`<svg xmlns="http://www.w3.org/2000/svg" width="1920" height="1080" viewBox="0 0 1920 1080"><style>text{font-family:'Microsoft YaHei','微软雅黑',sans-serif}</style><rect width="1920" height="1080" fill="white"/>`);
b.push(text('九人狼人杀 · 身份配置',80,94,52,blue,700));
b.push(text('预言家 1 人  ·  女巫 1 人  ·  猎人 1 人  ·  平民 3 人  ·  狼人 3 人',82,150,26,grey));
b.push(`<path d="M80 187H1840" stroke="${blue}" stroke-width="4"/>`);
b.push('<rect x="80" y="225" width="1160" height="700" rx="18" fill="#F1F6FC"/><rect x="1272" y="225" width="568" height="700" rx="18" fill="#FAF0F1"/>');
b.push(text('好人阵营',120,285,34,blue,700),text('6 人',1198,285,32,blue,700,'end'));
b.push(text('狼人阵营',1312,285,34,red,700),text('3 人',1798,285,32,red,700,'end'));
const roles=[
 {name:'预言家',icon:'eye',cx:300,lines:['每晚可查验一名玩家','得知其是好人还是狼人']},
 {name:'女巫',icon:'flask-conical',cx:660,lines:['拥有一瓶解药','和一瓶毒药']},
 {name:'猎人',icon:'crosshair',cx:1020,lines:['符合条件出局时','可开枪带走一人']}
];
for(const r of roles){
 b.push(await icon(r.icon,r.cx-49,350,98,blue));
 b.push(text(r.name+' × 1',r.cx,500,34,blue,700,'middle'));
 r.lines.forEach((l,i)=>b.push(text(l,r.cx,550+i*38,24,ink,400,'middle')));
}
b.push('<path d="M120 638H1200" stroke="#CCDCEC" stroke-width="2"/>');
b.push(text('平民 × 3',120,723,31,blue,700),text('没有特殊能力',120,771,24,ink),text('依靠发言与投票判断',120,811,24,ink));
for(const cx of [600,820,1040]){b.push(await icon('user-round',cx-40,704,80,blue));b.push(text('平民',cx,835,25,blue,600,'middle'));}
for(const cx of [1380,1556,1732])b.push(await icon('wolf-head',cx-54,353,108,red));
b.push(text('狼人 × 3',1556,535,38,red,700,'middle'));
b.push(text('夜间共同选择袭击目标',1556,600,25,ink,400,'middle'),text('白天隐藏身份、参与讨论',1556,642,25,ink,400,'middle'));
b.push('<path d="M1312 700H1800" stroke="#E5C7CC" stroke-width="2"/>');
b.push(text('同属一个阵营',1556,770,26,red,600,'middle'),text('通常互相知道身份',1556,815,25,ink,400,'middle'));
b.push(text('所有角色都会发言和投票；公开声明的身份不一定是真实身份。',80,985,27,blue,600));
b.push(text('注：女巫用药限制、猎人技能触发条件等，以具体对局规则为准。',80,1034,19,grey));
b.push('</svg>');
const svg=b.join('\n');
await fs.writeFile(out+'.svg',svg,'utf8');
await sharp(Buffer.from(svg)).resize(2880,1620).png().toFile(out+'.png');
await fs.writeFile(out+'_来源.txt',`角色配置：按用户指定的9人配置绘制。\n技能参考：https://www.langrensha.net/strategy/2021050801.html\n图标：Lucide（ISC License），https://lucide.dev/license\n预言家：eye；女巫：flask-conical；猎人：crosshair；平民：user-round。\n狼人图标：Wolf Head by Lorc，Game-icons.net，CC BY 3.0。\nhttps://game-icons.net/1x1/lorc/wolf-head.html\nhttps://creativecommons.org/licenses/by/3.0/\n所有图标经统一配色与尺寸调整。图为SVG排版绘制，未使用AI生成图片。\n`,'utf8');
console.log(JSON.stringify({png:out+'.png',svg:out+'.svg',size:[2880,1620]}));
