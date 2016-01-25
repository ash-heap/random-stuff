" $HOME/.vimrc
"
" This is the main vimrc file.  It is installation independent and suitable for
" syncing across machines regardless of the software installed on them.
"
"
" Last Edited: Monday, August 11, 2014.223 19:05:17
"              Michael R. Shannon <mrshannon.aerospace@gmail.com>
"

let g:vimpager_less_mode = 0
let g:vimpager_passthrough = 0


let g:syntastic_c_include_dirs = ['/opt/microchip/mplabc18/v3.40/h']

" Vundle
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
set nocompatible
filetype off

" Set the runtime path to include Vundle and initialize
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()
" Alternative, pass a path where Vundle should install plugins:
"   call cundle#begin('~/some/path/here')


" NOTE: If only a package name is given it will use vim-scripts.org.  If a
" username/package is given then it will be downloaded from GitHub.

" --- Vundle ---
" Add Vundle itself to Vundle.
" https://github.com/gmarik/Vundle.vim
Plugin 'gmarik/Vundle.vim'

Plugin 'rkulla/pydiction'

" --- Skittles Berry ---
" Skittles Berry colorscheme.
" https://github.com/shawncplus/skittles_berry
Plugin 'shawncplus/skittles_berry'

" --- Bufkill ---
" Adds an alternative to the :bun/:bd/:bw that leaves the window open.  The
" alternatives provided by this plugin are: :BUN/:BD/:BW.
" https://github.com/vim-scripts/bufkill.vim
Plugin 'bufkill.vim'

" --- CamelCaseMotion ---
" Adds the ',w', ',b', and ',e' normal mode commands that do the same thing as
" 'w', 'b', and 'e' but works on CamelCase and under_score style words.
" https://github.com/bkad/CamelCaseMotion
Plugin 'bkad/CamelCaseMotion'

" --- Gundo ---
" Graphical undo for Vim.
" https://github.com/sjl/gundo.vim
Plugin 'sjl/gundo.vim'

" --- vim-matlab ---
" Adds MATLAB syntax to Vim.
" http://www.mathworks.com/matlabcentral/fileexchange/21798-editing-matlab-files-in-vim
Plugin 'file:///home/mrshannon/projects/vim-matlab'

" --- NERD Tree ---
" Enhanced file browser for Vim.
" https://github.com/scrooloose/nerdtree
Plugin 'scrooloose/nerdtree'

" --- vim-pandoc ---
" https://github.com/vim-pandoc/vim-pandoc
Plugin 'vim-pandoc/vim-pandoc'

" --- vim-pandoc-syntax ---
" Requires vim-pandoc
" https://github.com/vim-pandoc/vim-pandoc-syntax
Plugin 'vim-pandoc/vim-pandoc-syntax'

" --- vim-rails ---
" Ruby on Rails integration into Vim.
" https://github.com/tpope/vim-rails
Plugin 'tpope/vim-rails'

" --- vim-scala ---
" Scala integration into Vim.
" https://github.com/scala/scala-dist/tree/master/tool-support/src/vim
Plugin 'scala/scala-dist', {'name': 'vim-scala', 'rdp': 'tool-support/src/vim/'}

" --- vim-surround ---
" Add, change, or remove soundings such as html tags, quotes, parentheses,
" brackets, and braces.
" https://github.com/tpope/vim-surround
Plugin 'tpope/vim-surround'

" --- Tabular ---
" Adds advanced alignment capability to Vim.
" http://vimcasts.org/episodes/aligning-text-with-tabular-vim/A
" https://github.com/vim-scripts/Tabular
Plugin 'tabular'

" --- Tagbar ---
" Source code browser for Vim.
" http://majutsushi.github.io/tagbar/
" https://github.com/majutsushi/tagbar
Plugin 'majutsushi/tagbar'

" --- TComment ---
" Toggle line comments with gc{motion} or gcc for the current line.
" https://github.com/tomtom/tcomment_vim
Plugin 'tomtom/tcomment_vim'


" --- TODO ---
Plugin 'mattn/emmet-vim'


" --- TODO ---
Plugin 'scrooloose/syntastic'



Plugin 'Shougo/vimproc.vim'


Plugin 'eagletmt/ghcmod-vim'


Plugin 'Conque-GDB'

Plugin 'rainbow_parentheses.vim'

Plugin 'Yggdroot/indentLine'


" All plugins must be added before the following line.
call vundle#end()

" Re-enable filetype.
filetype plugin indent on

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""



let $PYTHONPATH='/usr/lib/python3.4/site-packages'


let g:pydiction_location = '/home/mrshannon/.vim/bundle/pydiction/complete-dict'






" Set indentLine color
let g:indentLine_color_term = 239



if &term =~ '256color'
  " disable Background Color Erase (BCE) so that color schemes
  " render properly when inside 256-color tmux and GNU screen.
  " see also http://snk.tuxfamily.org/log/vim-256color-bce.html
  set t_ut=
endif



""" General
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

" Do not use vi compatibility mode.  (this is really a noop on modern vim as it
" is the default)
set nocompatible


" Turn on status bar.
set laststatus=2


" Allow leaving a buffer without saving it - retaining it's information.
set hidden


" Turn off spell check and enable the leader key + s to toggle spell check.
set nospell
nmap <leader>s  :set spell!<CR>


" Turn on line numbers and set <Leader>n to goggle them.
set number
nmap <Leader>n  :setlocal number!<CR>


" Show matching parenthesis.
set showmatch


" Search in real time.
set incsearch


" Make searches case insensitive except when pattern contains a capital.
"set ignorecase

" Turn on line and column readout.
set ruler


" Keep space between the cursor and edge of the screen.
set scrolloff=4


" Set vim to act like the Linux terminal with file names.
set wildmode=longest,list  


" Do not highlight search results.
set nohlsearch


" Display the input of incomplete commands as they are typed.
set showcmd


" Set printing options.
set printoptions=paper:letter,left:10pc,right:10pc,top:10pc,bottom:10pc
set printfont=courier:h8
set printdevice=Photosmart_C4400


" Turn on display of tabs and end of line characters.
set list
set listchars=tab:▸\ ,eol:¬
nmap <Leader>l  :set list!<CR>


" Set the arrow keys to move by visual line and not by actual line.
nmap <Down> gj
nmap <Up> gk
vmap <Down> gj
vmap <Up> gk


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

" Use %% to expand to the current directory on the command prompt.
"
"   http://vimcasts.org/episodes/the-edit-command/
"
cnoremap %% <C-R>=fnameescape(expand('%:h')).'/'<CR>





""" Color Scheme
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
try
    if $TERM != "xterm-256color" && !has("gui_running")
        echoerr "Not 256 color compatible!"
    endif
    colorscheme skittles_berry
    highlight NonText ctermfg=235 guifg=#262626 " Mutes the end of line character
catch
    try
        colorscheme desert
    catch
        colorscheme default
    endtry
    highlight NonText ctermfg=DarkGrey guifg=#262626 " Mutes the end of line character
endtry
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""




""" Buffers
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

nmap <C-b> :bdelete<CR>
nmap <C-n> :bprevious<CR>
nmap <C-m> :bnext<CR>

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""




""" Tabs
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

" Switch tabs.
nmap <C-j> gT
nmap <C-k> gt

" Move tabs.
nmap <C-h> :tabmove -1<CR>
nmap <C-l> :tabmove +1<CR>

" Open and close tabs.
nmap <C-y> :tabclose<CR>
nmap <C-u> :tabnew<CR>

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""






""" Abbreviations
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

" set "@dts@" to expand to the date and time.
ab <expr> @dts@ strftime("%A, %B %d, %Y.%j %H:%M:%S")

" set "@dt@" to expand to the date.
ab <expr> @dt@ strftime("%m/%d/%Y.%j")

" set "@me@" to expand to my name.
ab @me@ Michael R. Shannon

" set "@fn@" to expand to the current filename.
ab <expr> @fn@ expand('%:t')

" set "@fp@" to expand to the relative path of the current file.
ab <expr> @fp@ expand('%')

" set "@fpa@" to expand to the full path of the current file.
ab <expr> @fpa@ expand('%:p')

" set "@dn@" to expand to the relative path of the directory containing the
" current file
ab <expr> @dn@ fnameescape(expand('%:h')).'/'

" set "@dna@" to expand to the full path of the directory containing the current
" file
ab <expr> @dna@ fnameescape(expand('%:p:')).'/'

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""




""" Tab (characer/key) Settings
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

" Keep same level of indentation from previous line.
" NOTE: This does not interfere with filetype indentation settings.
set autoindent

" Make shift-tab work as backwards tab.
imap <S-Tab> <Esc><<i


" Make tab key work in command mode.
nmap <S-Tab> <<
nmap <Tab> >>


" Make tab key work in visual mode.
vmap <S-Tab> <gv
vmap <Tab> >gv


" Use tabs or spaces  Set <Leader>+t to toggle locally and <Leader>+T to toggle
" globally.
set expandtab
nmap <Leader>t  :setlocal expandtab!<CR>
nmap <Leader>T  :set expandtab!<CR>


" Tab width.  Set default value and define commands to list and set the tab
" width.
" 
"   :Ltab               # List the current tab width (and expandtab state)
"   :Stab <tab width>   # Set the tab width
"
command! -nargs=1 Stab call SetTabWidth(<f-args>)
function SetTabWidth(spaces)
    let &l:tabstop=a:spaces
    let &l:softtabstop=a:spaces
    let &l:shiftwidth=a:spaces
endfunction
command! -nargs=0 Ltab call ListTabWidth()
function ListTabWidth()
    echon 'tabstop='.&l:tabstop
    echon '  softtabstop='.&l:softtabstop
    echon '  shiftwidth='.&l:shiftwidth
    if &l:expandtab
        echon '  (spaces)'
    else
        echon '  (tabs)'
    endif
endfunction
call SetTabWidth(4)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""








""" Line Wrap/Break Settings
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

" Enable soft wrapping.
set wrap

" Don't softbreak lines in the middle of words.
" NOTE: if "list" is enabled then this will not work.
set linebreak

" Show "..." on soft linebreaks.
set showbreak=...

" Set the default textwidth.  This is overwidden for some filetypes in the
" FileType section.
set textwidth=78

" Use the "par" program for gq formating and vim for gw.
set formatprg=par\ -w78rq

" Set the default format options for formating paragraphs.
if has("autocmd")
    au BufNewFile,BufRead * setlocal formatoptions=roqlj2
endif

" Set <Leader>p to toggle between automatic linebreaks and manual line breaks.
nmap <Leader>p :call ToggleAutoWrap()<CR>
function ToggleAutoWrap()
    if (&l:formatoptions != "tacroqlj2")
        let &l:formatoptions="tacroqlj2"
        echo "Set to autowrap."
    else
        let &l:formatoptions="roqlj2"
        echo "Set to manual wrap."
    endif
endfunction
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""





" Allow copying to and pasting from the Xserver register.
" These use the escape key as a modifier key.
nmap <Esc>p "+p
nmap <Esc>P "+P
nmap <Esc>y "+yy
nmap <Esc>d "+dd
vmap <Esc>p "+p
vmap <Esc>P "+P
vmap <Esc>y "+y
vmap <Esc>d "+d










""" FileType Settings
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

syntax on
filetype plugin indent on

if has("autocmd")


    """ turn on rainbow parentheses
    au VimEnter * RainbowParenthesesToggle
    au Syntax * RainbowParenthesesLoadRound
    au Syntax * RainbowParenthesesLoadSquare
    au Syntax * RainbowParenthesesLoadBraces

    " NOTE: tw  = textwdith
    "       ts  = tabstop
    "       sts = softtabstop
    "       sw  = shiftwidth
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    " C
    autocmd FileType c setlocal tw=72 ts=4 sts=4 sw=4 expandtab colorcolumn=+1,+2,+3,+4,+5,+6,+7

    " C++
    autocmd BufNewFile,BufRead *.template setfiletype cpp
    autocmd FileType cpp setlocal tw=72 ts=4 sts=4 sw=4 expandtab colorcolumn=+1,+2,+3,+4,+5,+6,+7

    " Haskell
    autocmd FileType haskell setlocal tw=72 ts=4 sts=4 sw=4 expandtab colorcolumn=+1,+2,+3,+4,+5,+6,+7

    " Java
    autocmd FileType java setlocal tw=72 ts=4 sts=4 sw=4 expandtab colorcolumn=+1,+2,+3,+4,+5,+6,+7

    " JavaScript
    autocmd FileType javascript setlocal tw=72 ts=4 sts=4 sw=4 expandtab colorcolumn=+1,+2,+3,+4,+5,+6,+7

    " Makefile
    autocmd FileType make setlocal tw=72 ts=4 sts=4 sw=4 noexpandtab colorcolumn=+1,+2,+3,+4,+5,+6,+7

    " Markdown
    autocmd FileType markdown setlocal tw=78 ts=2 sts=2 sw=2 expandtab
    autocmd FileType pandoc setlocal tw=78 ts=2 sts=2 sw=2 expandtab cole=0

    " MATLAB
    autocmd FileType matlab setlocal tw=72 ts=4 sts=4 sw=4 expandtab colorcolumn=+1,+2,+3,+4,+5,+6,+7

    " Python
    autocmd FileType python setlocal tw=72 ts=4 sts=4 sw=4 expandtab colorcolumn=+1,+2,+3,+4,+5,+6,+7

    " Ruby
    autocmd FileType ruby setlocal tw=72 ts=2 sts=2 sw=2 expandtab colorcolumn=+1,+2,+3,+4,+5,+6,+7

    " Scala
    autocmd FileType scala setlocal tw=72 ts=2 sts=2 sw=2 expandtab colorcolumn=+1,+2,+3,+4,+5,+6,+7

    " txt2tags
    autocmd BufNewFile,BufRead *.t2t setfiletype txt2tags
    autocmd FileType txt2tags setlocal tw=78 ts=2 sts=2 sw=2 expandtab

    " LaTeX
    autocmd FileType plaintex setlocal tw=78 ts=4 sts=4 sw=4 expandtab
    autocmd FileType tex setlocal tw=78 ts=4 sts=4 sw=4 expandtab

endif
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""




let g:tagbar_type_scala = {
    \ 'ctagstype' : 'Scala',
    \ 'kinds'     : [
        \ 'p:packages:1',
        \ 'V:values',
        \ 'v:variables',
        \ 'T:types',
        \ 't:traits',
        \ 'o:objects',
        \ 'a:aclasses',
        \ 'c:classes',
        \ 'r:cclasses',
        \ 'm:methods'
    \ ]
\ }









nmap <leader>C viw :s/\%V_\(.\)/\u\1/ge<CR> :s/\%V\(.\)/\u\1/e<CR> `<
nmap <leader>c viw :s/\%V_\(.\)/\u\1/ge<CR> :s/\%V\(.\)/\l\1/e<CR> `<
nmap <leader>u viw :s/\%V\(\w\)\(\u\)/\1_\2/ge<CR> :s/\V\(\u\)/\l\1/ge<CR> `<





" Holds settings relating to plugins.  This function will be called after Vim
" startup so that available plugins can be evaluated.
"
"
" http://stackoverflow.com/questions/5010162/if-existscommand-fails-on-startup-using-pathogen
"
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
autocmd VimEnter * :call Plugins()
function Plugins()


    " surround.vim
    " 
    "   This plugin makes changing/adding/removing surrounding marks easier.
    "
    "   https://github.com/tpope/vim-surround
    "   http://www.vim.org/scripts/script.php?script_id=1697
    "
    "   To change surrounding characters/tags:
    "       cs{old surround char}{new surround char/tag}
    "
    "   If changing from an html/xml tag then use:
    "       cst{new surround char/tag}
    "
    "   To add surrounding characters/tags:
    "       ys{motion}{new surround char/tag}
    "
    "   NOTE: ) is for parentheses with surrounding spaces.  For tight
    "         parentheses use b instead.
    "
    "   To delete surrounding char/tags use:
    "       ds{surround char}
    "
    "   NOTE: Use t as the <surround char> to remove a tag.
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""






    " tocomment.vim
    "
    " This plugin makes it easy to comment out a block of code.
    "
    "   http://www.vim.org/scripts/script.php?script_id=1173
    "   https://github.com/tomtom/tcomment_vim
    "
    "
    "   Toggle comment in normal mode:
    "       gc<motion>
    "
    "   Toggle comment on current line:
    "       gcc
    "
    "   Toggle comment of selected block in visual mode:
    "       gc
    "
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""



    " bufkill.vim
    "   
    "   Close buffers while leaving the window open.
    "
    "   http://www.vim.org/scripts/script.php?script_id=1147
    "   https://github.com/vim-scripts/bufkill.vim
    "
    "
    "   Unload a buffer:
    "       :bun - standard vim command
    "       :BUN - keep window open
    "
    "   Delete a buffer:
    "       :bd - standard vim command
    "       :BD - keep window open
    "
    "   Wipe out buffer:
    "       :bw - standard vim command
    "       :BW - keep window open
    "
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""



    " camelcasemotion.vim
    "
    "   Motion through CamelCase and underscore_notation.
    "
    "   http://www.vim.org/scripts/script.php?script_id=1905
    "
    "
    "   Move to beginning of next word:
    "       w   - standard vim command
    "       ,w  - move through CamelCase and underscore_notation
    "
    "   Move to end of current/next word:
    "       e   - standard vim command
    "       ,e  - move through CamelCase and underscore_notation
    "
    "   Move to beginning of current/previous word:
    "       b   - standard vim command
    "       ,b  - move through Camelcase and underscore_notation
    "
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    " Use 'W', 'B', and 'E'. instead of ',w', ',b', and ',e' for the CamelCaseMotion
    " plugin.
    " if exists(":CamelCaseMotion_w")
        map <S-W> <Plug>CamelCaseMotion_w
        map <S-B> <Plug>CamelCaseMotion_b
        map <S-E> <Plug>CamelCaseMotion_e
    " endif
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""



    " NERDTree.vim
    " 
    "
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    if exists(":NERDTree")
        nmap <C-p> :NERDTreeToggle<CR>
    endif
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""




    " Tagbar
    " 
    "
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    if exists(":Tagbar")
        nmap <C-i> :TagbarToggle<CR>
        nmap <C-o> :TagbarOpenAutoClose<CR>
        let g:tagbar_width=45
    endif
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""



    " Gundo
    "
    "
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    if exists(":Gundo")
        nmap <C-g> :GundoToggle<CR>
    endif
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""



    " Tabularize
    "
    " http://vimcasts.org/episodes/aligning-text-with-tabular-vim/
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    if exists(":Tabularize")

        nmap <Leader>a= :Tabularize /=<CR>
        vmap <Leader>a= :Tabularize /=<CR>

        nmap <Leader>a=s :Tabularize /=\zs<CR>
        vmap <Leader>a=s :Tabularize /=\zs<CR>

        nmap <Leader>a, :Tabularize /,<CR>
        vmap <Leader>a, :Tabularize /,<CR>,

        nmap <Leader>a,s :Tabularize /,\zs<CR>
        vmap <Leader>a,s :Tabularize /,\zs<CR>

        nmap <Leader>a\| :Tabularize /\|<CR>
        vmap <Leader>a\| :Tabularize /\|<CR>

        nmap <Leader>a\|s :Tabularize /\|\zs<CR>
        vmap <Leader>a\|s :Tabularize /\|\zs<CR>

        nmap <Leader>a: :Tabularize /:<CR>
        vmap <Leader>a: :Tabularize /:<CR>

        nmap <Leader>a:s :Tabularize /:\zs<CR>
        vmap <Leader>a:s :Tabularize /:\zs<CR>

    endif
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""



    " Emmet
    "
    "
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    if exists(":Emmet")
        let g:user_emmet_leader_key='<C-Z>'
    endif
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""



    " Syntastic
    "
    "
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    if exists(":SyntasticToggleMode")
        nmap <C-x> :SyntasticToggleMode<CR>
        "let g:syntastic_haskell_checkers = ['hlint']
    endif
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""



    " This is a Haskell configuration for the Vim Tagbar plugin that uses
    " lushtags.
    "
    " Tagbar can be found at:
    " http://www.vim.org/scripts/script.php?script_id=3465
    " http://majutsushi.github.com/tagbar/
    "
    " Paste this in to your vimrc file
    " OR copy this file into your .vim/plugin directory
    " OR load it from your vimrc file by adding a line like:
    "
    " source /path/to/tagbar-haskell.vim

    if executable('lushtags')
        let g:tagbar_type_haskell = {
            \ 'ctagsbin' : 'lushtags',
            \ 'ctagsargs' : '--ignore-parse-error --',
            \ 'kinds' : [
                \ 'm:module:0',
                \ 'e:exports:1',
                \ 'i:imports:1',
                \ 't:declarations:0',
                \ 'd:declarations:1',
                \ 'n:declarations:1',
                \ 'f:functions:0',
                \ 'c:constructors:0'
            \ ],
            \ 'sro' : '.',
            \ 'kind2scope' : {
                \ 'd' : 'data',
                \ 'n' : 'newtype',
                \ 'c' : 'constructor',
                \ 't' : 'type'
            \ },
            \ 'scope2kind' : {
                \ 'data' : 'd',
                \ 'newtype' : 'n',
                \ 'constructor' : 'c',
                \ 'type' : 't'
            \ }
        \ }
    endif


endfunction
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
