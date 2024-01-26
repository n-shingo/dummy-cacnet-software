# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 09:07:35 2024

@author: shingo
"""
import cv2
import numpy as np
import math
from math import pi, sin, cos
from PIL import Image, ImageDraw, ImageFont, ImageFilter

class DrawTool:

    __warning_pnts = None
    def warning_icon( img, xy, size, base_color=(251, 232, 69), str_color=(0,0,0) ):
        
        '''
        警告アイコンを描画する
        '''
        rt3 = 1.7320508
        r = 0.1
        if DrawTool.__warning_pnts is None:
            step_angle = 30
            pnts = []
        
            # 高さ１の角丸三角形描画点
            cen = [ ( r+(1-2*r)/rt3, r ), ( r+2*(1-2*r)/rt3, 1-r ), (r, 1-r)]
            for ang in range( 210, 331, step_angle ):
                ang *= pi / 180.0
                pnts.append( ( cen[0][0] + r*cos(ang), cen[0][1] + r*sin(ang) ) )
            
            for ang in range( -30, 91, step_angle ):
                ang *= pi / 180.0
                pnts.append( ( cen[1][0] + r*cos(ang), cen[2][1] + r*sin(ang) ) )
                
            for ang in range( 90, 211, step_angle ):
                ang *= pi / 180.0
                pnts.append( ( cen[2][0] + r*cos(ang), cen[2][1] + r*sin(ang) ) )
            
            DrawTool.__warning_pnts = pnts

        # 描画画像サイズ        
        mgn = 16
        icon_size = (2*(1-2*r)/rt3+2*r, 1.0)  # 高さ1のiconサイズ
        image_size = (int(size*icon_size[0]+2*mgn), int( size*icon_size[1]+2*mgn) ) # 処理画像のサイズ

        # 四角にエクスクラメーションマーク描画
        icon_img = Image.new( 'RGB', image_size, color=base_color )
        draw = ImageDraw.Draw(icon_img)
        font_size = int(0.8*size)+1
        font=ImageFont.truetype('meiryob.ttc', font_size, index=2)
        draw.text( (mgn + size*0.42, mgn + size*0.04), '!', fill =str_color,  font=font )

        # Mask 画像作成
        icon_mask = Image.new( 'L', image_size )
        pnts = [ (mgn+px*size, mgn+py*size) for px, py in DrawTool.__warning_pnts ]
        draw = ImageDraw.Draw(icon_mask)
        draw.polygon(pnts, fill=255)
        icon_mask = icon_mask.filter(ImageFilter.GaussianBlur(radius=0.8))
        
        # 合成
        x, y = xy
        img.paste( icon_img, (int(x-mgn), int(y-mgn)), icon_mask)
    
    
    def information_icon( img, xy, size, base_color=(38, 84, 157), str_color=(255,255,255) ):
        
        '''
        Infoアイコンを描画する
        '''

        # 描画画像サイズ        
        mgn = 16
        image_size = (int(size+2*mgn), int( size+2*mgn) ) # 処理画像のサイズ

        # 真ん中にエクスクラメーションマーク描画
        icon_img = Image.new( 'RGB', image_size, color=base_color )
        draw = ImageDraw.Draw(icon_img)
        font_size = int(0.75*size)+1
        font=ImageFont.truetype('meiryob.ttc', font_size, index=2)
        draw.text( (mgn + size*0.42, mgn + size*0.04), 'i', fill =str_color,  font=font )

        # Mask 画像作成
        icon_mask = Image.new( 'L', image_size )
        draw = ImageDraw.Draw(icon_mask)
        draw.ellipse( (mgn, mgn, int(mgn+size), int(mgn+size) ), fill=255)
        icon_mask = icon_mask.filter(ImageFilter.GaussianBlur(radius=0.8))
        
        # 合成
        x, y = xy
        img.paste( icon_img, (int(x-mgn), int(y-mgn)), icon_mask)
    
    
    def direct_gadget( img, xy, direct_dif, width, threshold ):
        
        '''
        方向ガジェットを描画
        '''
        
        # 全体サイズ
        height = 0.357 * width

        # 描画領域
        w = int(width)
        h = int(height)
        x = int( xy[0] )
        y = int( xy[1] )
        
        # 描画領域切り出し
        draw_img = img[y:y+h, x:x+w, :].copy()

        # インジケータ描画
        idc_size = 0.8*height
        idc_cx = 0.1*height+idc_size/2
        idc_cy = height/2
        DrawTool.translucent_rect(draw_img, 0, 0, w, h, color=(0,0,0))
        DrawTool.direct_indicator(draw_img, (idc_cx, idc_cy), direct_dif, idc_size, threshold)
        
        # PIL で処理
        pil_img = DrawTool.to_pil_image(draw_img)
        
        # 文字 'Direction'
        msg = 'DIRECTION'
        font_size = 0.2*height
        font_x = height
        font_y = height/3
        DrawTool.text(pil_img, (font_x, font_y), msg, font_size, color=(255,255,255), anchor='lm')
        
        # 文字 '(x%, y%)
        if direct_dif is None:
            msg = '( None, None )'
        else:
            val_x = -direct_dif[0]*100
            val_y = direct_dif[1]*100
            msg = '({0:>4.1f}%, {1:>4.1f}%)'.format( val_x, val_y )
        font_y = height*2/3
        DrawTool.text(pil_img, (font_x, font_y), msg, font_size, color=(255,255,255), anchor='lm')
        
        # cv2 imageに戻す
        draw_img = DrawTool.to_cv2_image(pil_img)
        
        # 元には貼り付け
        img[y:y+h, x:x+w, :] = draw_img
        
    
    def angle_gadget( img, xy, angle_dif, width, matched:bool ):
        
        '''
        角度ガジェットを描画
        '''
        
        # 全体サイズ
        height = 0.357 * width

        # 描画領域
        w = int(width)
        h = int(height)
        x = int( xy[0] )
        y = int( xy[1] )
        
        # 描画領域切り出し
        draw_img = img[y:y+h, x:x+w, :].copy()

        # インジケータ描画
        idc_size = 0.8*height
        idc_cx = 0.1*height+idc_size/2
        idc_cy = height/2
        DrawTool.translucent_rect(draw_img, 0, 0, w, h, color=(0,0,0))
        DrawTool.angle_indicator(draw_img, (idc_cx, idc_cy), angle_dif, idc_size, matched )
        
        # PIL で処理
        pil_img = DrawTool.to_pil_image(draw_img)
        
        # 文字 'Angle'
        msg = 'ANGLE'
        font_size = 0.2*height
        font_x = height
        font_y = height/3
        DrawTool.text(pil_img, (font_x, font_y), msg, font_size, color=(255,255,255), anchor='lm')
        
        # 文字 'val degree'
        if angle_dif is None:
            msg = 'None'
        else:
            val = angle_dif
            msg = '{0:>4.1f} degree'.format( val )
        font_y = height*2/3
        DrawTool.text(pil_img, (font_x, font_y), msg, font_size, color=(255,255,255), anchor='lm')
        
        # cv2 imageに戻す
        draw_img = DrawTool.to_cv2_image(pil_img)
        
        # 元には貼り付け
        img[y:y+h, x:x+w, :] = draw_img
        
    def zoom_gadget( img, xy, zoom_dif, width, matched:bool ):
        
        '''
        ZOMMガジェットを描画
        '''
        
        # 全体サイズ
        height = 0.357 * width

        # 描画領域
        w = int(width)
        h = int(height)
        x = int( xy[0] )
        y = int( xy[1] )
        
        # 描画領域切り出し
        draw_img = img[y:y+h, x:x+w, :].copy()

        # インジケータ描画
        idc_size = 0.8*height
        idc_cx = 0.1*height+idc_size/2
        idc_cy = height/2
        DrawTool.translucent_rect(draw_img, 0, 0, w, h, color=(0,0,0))
        DrawTool.zoom_indicator(draw_img, (idc_cx, idc_cy), zoom_dif, idc_size, matched )
        
        # PIL で処理
        pil_img = DrawTool.to_pil_image(draw_img)
        
        # 文字 'ZOOM'
        msg = 'ZOOM'
        font_size = 0.2*height
        font_x = height
        font_y = height/3
        DrawTool.text(pil_img, (font_x, font_y), msg, font_size, color=(255,255,255), anchor='lm')
        
        # 文字 'x AMP'
        if zoom_dif is None:
            msg = 'None'
        else:
            msg = 'x{0:>4.3f}'.format( math.e**zoom_dif )
        font_y = height*2/3
        DrawTool.text(pil_img, (font_x, font_y), msg, font_size, color=(255,255,255), anchor='lm')
        
        # cv2 imageに戻す
        draw_img = DrawTool.to_cv2_image(pil_img)
        
        # 元には貼り付け
        img[y:y+h, x:x+w, :] = draw_img
        
    @staticmethod
    def direct_indicator( img, cent_xy, direct_dif, size, threshold, maxval=0.08 ):
        
        # 整数化
        cx, cy = [ int(cv) for cv in cent_xy ]
        rad = int(size/2)
        in_rad = int(rad*threshold/maxval)
        
        # 色
        white = (255,255,255)
        red = (0,0,192)
        green = (0,192,0)
        
        # 位置を描画できる場合
        if direct_dif is not None:
            
            # 値取得
            dx, dy = direct_dif
            d = (dx*dx+dy*dy)**0.5
            matched = d < threshold
    
            # 位置
            r = min( max( d, -maxval ), +maxval )
            if d < 1.0e-10:
                dx, dy = 0, 0
            else:
                dx = int(r*dx/d * rad / maxval)
                dy = int(r*dy/d * rad / maxval)

            # 内側の円を塗りつぶし
            color = green if matched else red
            cv2.circle( img, (cx,cy), in_rad, color, -1, lineType=cv2.LINE_AA )
            
            # インジケータ針描画
            cv2.circle( img, (cx,cy), 3, white, -1, lineType=cv2.LINE_AA)
            cv2.rectangle( img, (cx-dx-3,cy-dy-3), (cx-dx+3,cy-dy+3), white, thickness=-1, lineType=cv2.LINE_AA )
            cv2.line( img, (cx, cy), (cx-dx, cy-dy), white, thickness=2, lineType=cv2.LINE_AA  )
        
        # インジケータ残り描画
        cv2.line( img, (cx-rad, cy), (cx+rad, cy), white, thickness=1, lineType=cv2.LINE_AA  )
        cv2.line( img, (cx, cy-rad), (cx, cy+rad), white, thickness=1, lineType=cv2.LINE_AA  )
        cv2.circle( img, (cx,cy), in_rad, white, thickness=1, lineType=cv2.LINE_AA )
        cv2.circle( img, (cx,cy), rad, white, thickness=2, lineType=cv2.LINE_AA )

    @staticmethod
    def angle_indicator( img, cent_xy, angle_dif, size, matched, amp=2.0 ):
        
        # 色
        white = (200,200,200)
        red = (0,0,192)
        green = (0,192,0)
        
        # サイズ、位置など整数値で取得
        cx, cy = [ int(cv) for cv in cent_xy ]
        rad = int(size/2)
        
        # 水平角度インジケータ描画
        if not angle_dif is None:
            color = green if matched else red
            cv2.ellipse(img, (cx,cy), (rad, rad), -amp*angle_dif, 0, 180, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.line( img, (cx-rad, cy), (cx+rad, cy), white, thickness=1, lineType=cv2.LINE_AA  )
        cv2.circle( img, (cx,cy), rad, white, thickness=2, lineType=cv2.LINE_AA )
        
    @staticmethod
    def zoom_indicator( img, cent_xy, zoom_dif, size, matched, max_val=0.5 ):
        
        # 色
        white = (200,200,200)
        red = (0,0,192)
        green = (0,192,0)

        # サイズ、位置など整数値で取得
        cx, cy = [ int(cv) for cv in cent_xy ]
        w, h = int(size), int(size*2/3)
        w2, h2 = w//2, h//2
        line_y = int(cy+0.08*h)
        main_scale_len = int(0.3*h) # 真ん中目盛り長さ
        sub_scale_len = int(0.15*h)  # 他目盛り長さ
        
        #### ズームインジケータ描画 ###

        # 目盛り描画
        cv2.line( img, (cx-w2, line_y), (cx+w2, line_y), white, thickness=1, lineType=cv2.LINE_AA )
        dy = main_scale_len//2
        cv2.line( img, (cx, line_y-dy), (cx, line_y+dy), white, thickness=1, lineType=cv2.LINE_AA )
        for i in range(1,4):
            dx = int(w2*i/4)
            dy = sub_scale_len//2
            cv2.line( img, (cx-dx, line_y+dy), (cx-dx, line_y-dy), white, thickness=1, lineType=cv2.LINE_AA )
            cv2.line( img, (cx+dx, line_y+dy), (cx+dx, line_y-dy), white, thickness=1, lineType=cv2.LINE_AA )

        # 外枠描画
        cv2.rectangle( img, (cx-w2, cy-h2), (cx+w2, cy+h2), white, thickness=2, lineType=cv2.LINE_AA )
    
        # 三角形▼描画
        if not zoom_dif is None:
            color = green if matched else red
            val = max( min( zoom_dif, max_val ), -max_val )
            px = cx+int(w2*val/max_val)
            pnts = [(px, line_y-0.2*h), (px+0.1*w, line_y-0.45*h), (px-0.1*w, line_y-0.45*h)]
            pnts = np.array( pnts ).astype('int').reshape( 1, 3, 2 )
            cv2.fillPoly( img, [pnts], color )
            cv2.line( img, (px, cy-int(0.9*h2)), (px, cy+int(0.9*h2)), color, thickness=2, lineType=cv2.LINE_AA )

    @staticmethod
    def text( img, xy, text, size, color, anchor=None ):
        '''
        IPL Image にテキスト描画
        '''
        #font=ImageFont.truetype('msgothic.ttc', int(size) )
        font=ImageFont.truetype('meiryob.ttc', int(size) )
        draw = ImageDraw.Draw(img)
        draw.text( xy, text, color, font=font, anchor=anchor )
        return img
    
    @staticmethod
    def fillrect( img, xyxy, fill ):
        
        '''
        IPL Image に矩形描画（アルファブレンド可）
        '''
        # 座標を整数化
        xyxy = [ int(v) for v in xyxy ]
        w = xyxy[2] - xyxy[0]
        h = xyxy[3] - xyxy[1]

        # 混ぜる画像作成
        block_img = Image.new( 'RGBA', (w,h), color = fill )
        org_img = img.crop(xyxy)

        # 混ぜる
        alpha = fill[3] if len(fill)==4 else 255
        block_img = Image.blend(org_img, block_img, alpha/255 )
        
        # もどす
        img.paste( block_img, xyxy[:2] )
    
    @staticmethod
    def translucent_rect( img, x, y, w, h, color, trans_rate=0.7 ):
        '''
         半透明な矩形を描く
        '''
        sub_img = img[y:y+h, x:x+w]
        blank = np.zeros((h, w, 3), dtype='uint8' )
        blank += np.array(color, dtype='uint8')
        rect = cv2.addWeighted(sub_img, 1-trans_rate, blank, trans_rate, 1.0)
        img[y:y+h, x:x+w] = rect
        
    @staticmethod
    def dashed_line(img, start_point, end_point, gap, linewidth, color):
        '''
        破線を描く
        参考 https://emotionexplorer.blog.fc2.com/blog-entry-8.html
    
        '''
        li = DrawTool.__lineList(start_point[0], start_point[1], end_point[0], end_point[1])
        fwd = start_point
        bwd = start_point
        j = 0
        for i, pt in enumerate(li):
            if i % gap == 0:
                bwd = pt
    
                if(j % 2):
                    cv2.line(img, fwd, bwd, color, linewidth, lineType=cv2.LINE_AA)
                fwd = bwd
                j += 1
        return img
    
    @staticmethod
    def __lineList(x1, y1, x2, y2):
        '''
        ブレゼンハムのアルゴリズム
        参考 https://emotionexplorer.blog.fc2.com/blog-entry-8.html
    
        '''
        line_lst = []
        step = 0
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dx > dy:
            if x1 > x2 :
                step = 0
                if y1 > y2:
                    step = 1
                else:
                    step = -1
                x1, x2 = x2, x1 # swap
                y1 = y2;
            else:
                if y1 < y2:
                    step = 1
                else:
                    step = -1
            line_lst.append((x1, y1))
            s = dx >> 1
            x1 += 1
            while (x1 <= x2):
                s -= dy
                if s < 0:
                    s += dx
                    y1 += step
                line_lst.append((x1, y1))
                x1 += 1
        else:
            if y1 > y2:
                if x1 > x2:
                    step = 1
                else:
                    step = -1
               
                y1, y2 = y2, y1 # swap
                x1 = x2
            else:
                if x1 < x2:
                    step = 1
                else:
                    step = -1
            line_lst.append((x1, y1))
            s = dy >> 1
            y1 += 1
            while y1 <= y2:
                s -= dx
                if s < 0:
                    s += dy
                    x1 += step
                line_lst.append((x1, y1))
                y1 += 1
        return  line_lst
    
    
    @staticmethod
    def to_pil_image( cv2_img: np.ndarray ):
        return Image.fromarray(cv2_img[:, :, [2, 1, 0]]).convert('RGBA') # BGR->RGBA
    
    @staticmethod
    def to_cv2_image( pil_img: Image ):
        return np.array(pil_img)[:, :, [2, 1, 0]]