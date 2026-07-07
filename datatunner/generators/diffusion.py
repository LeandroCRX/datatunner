"""
Stable Diffusion para geração de imagens sintéticas
"""

from typing import List, Optional, Union, Dict, Tuple
import numpy as np
from pathlib import Path
import warnings
from PIL import Image

try:
    import torch
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    DIFFUSION_AVAILABLE = True
except ImportError:
    DIFFUSION_AVAILABLE = False
    warnings.warn(
        "Bibliotecas de diffusion não encontradas. "
        "Instale com: pip install diffusers transformers accelerate",
        ImportWarning
    )

from datatunner.generators.base import BaseSyntheticGenerator


class StableDiffusionGenerator(BaseSyntheticGenerator):
    """
    Gerador de imagens usando Stable Diffusion
    
    Usa modelos pré-treinados de Stable Diffusion para gerar
    imagens sintéticas a partir de prompts de texto.
    """
    
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-2-1",
        device: str = "cuda",
        dtype: str = "float16",
        random_seed: int = 42,
        use_safetensors: bool = True
    ):
        """
        Args:
            model_id: ID do modelo HuggingFace (ex: "stabilityai/stable-diffusion-2-1")
            device: Dispositivo (cuda, cpu, mps)
            dtype: Tipo de dados (float16, float32)
            random_seed: Seed para reprodutibilidade
            use_safetensors: Se deve usar safetensors
        """
        super().__init__(random_seed)
        
        if not DIFFUSION_AVAILABLE:
            raise ImportError(
                "Bibliotecas de diffusion não instaladas. "
                "Instale com: pip install diffusers transformers accelerate"
            )
        
        self.generator_name = "StableDiffusion"
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Configurar dtype
        if dtype == "float16" and self.device == "cuda":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        
        self.pipe = None
        self.generator = torch.Generator(device=self.device).manual_seed(random_seed)
        
        print(f"Inicializando Stable Diffusion: {model_id}")
        print(f"Dispositivo: {self.device}, dtype: {dtype}")
        
        self._load_model()
    
    def _load_model(self):
        """Carrega o modelo Stable Diffusion"""
        try:
            # Carregar pipeline
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                use_safetensors=True
            )
            
            # Otimizações
            if self.device == "cuda":
                # Usar DPM Solver para geração mais rápida
                self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    self.pipe.scheduler.config
                )
                
                # Habilitar attention slicing para economizar memória
                self.pipe.enable_attention_slicing()
                
                # Habilitar VAE tiling se disponível
                try:
                    self.pipe.enable_vae_tiling()
                except AttributeError:
                    pass
            
            self.pipe = self.pipe.to(self.device)
            
            # Desabilitar safety checker para geração mais rápida (opcional)
            # self.pipe.safety_checker = None
            
            print("✅ Stable Diffusion carregado com sucesso!")
            
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar Stable Diffusion: {e}")
    
    def fit(
        self,
        data: List[str],
        labels: Optional[np.ndarray] = None
    ):
        """
        Para Stable Diffusion, 'fit' apenas armazena prompts de referência
        
        Args:
            data: Lista de prompts de texto
            labels: Labels (opcional)
        """
        self.reference_prompts = data
        self.labels = labels
        print(f"✅ {len(data)} prompts de referência armazenados")
    
    def generate(
        self,
        n_samples: int,
        prompts: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        negative_prompt: Optional[str] = None,
        return_images: bool = False
    ) -> Union[List[str], List[Image.Image]]:
        """
        Gera imagens sintéticas usando Stable Diffusion
        
        Args:
            n_samples: Número de imagens a gerar
            prompts: Lista de prompts (se None, usa prompts aleatórios)
            output_dir: Diretório para salvar imagens
            num_inference_steps: Número de steps de inferência (25-50)
            guidance_scale: Escala de guidance (7.0-12.0)
            height: Altura da imagem (múltiplo de 8)
            width: Largura da imagem (múltiplo de 8)
            negative_prompt: Prompt negativo (o que evitar)
            return_images: Se deve retornar objetos Image ao invés de paths
            
        Returns:
            Lista de caminhos de imagens ou objetos Image
        """
        if self.pipe is None:
            raise ValueError("Modelo não carregado")
        
        # Usar prompts fornecidos ou gerar aleatórios
        if prompts is None:
            if hasattr(self, 'reference_prompts'):
                # Selecionar prompts aleatórios dos de referência
                prompts = np.random.choice(
                    self.reference_prompts,
                    size=n_samples,
                    replace=True
                ).tolist()
            else:
                raise ValueError(
                    "Nenhum prompt fornecido. "
                    "Use fit() com prompts ou passe prompts para generate()"
                )
        
        # Se apenas um prompt, replicar
        if isinstance(prompts, str):
            prompts = [prompts] * n_samples
        elif len(prompts) < n_samples:
            # Repetir prompts se necessário
            prompts = prompts * (n_samples // len(prompts) + 1)
            prompts = prompts[:n_samples]
        
        generated_images = []
        image_paths = []
        
        # Criar diretório de saída se necessário
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Gerando {n_samples} imagens com Stable Diffusion...")
        
        for i, prompt in enumerate(prompts):
            print(f"[{i+1}/{n_samples}] Gerando: '{prompt[:50]}...'")
            
            # Gerar imagem
            with torch.no_grad():
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    generator=self.generator
                )
            
            image = result.images[0]
            
            # Salvar ou retornar
            if output_dir:
                img_path = output_path / f"generated_{i:04d}.png"
                image.save(img_path)
                image_paths.append(str(img_path))
            
            if return_images or not output_dir:
                generated_images.append(image)
        
        print(f"✅ {n_samples} imagens geradas!")
        
        if return_images:
            return generated_images
        elif output_dir:
            return image_paths
        else:
            return generated_images
    
    def generate_from_class_prompts(
        self,
        class_prompts: Dict[int, str],
        n_samples_per_class: int,
        output_dir: str,
        **kwargs
    ) -> Tuple[List[str], List[int]]:
        """
        Gera imagens balanceadas para cada classe
        
        Args:
            class_prompts: Dicionário {class_id: prompt}
            n_samples_per_class: Número de amostras por classe
            output_dir: Diretório de saída
            **kwargs: Argumentos adicionais para generate()
            
        Returns:
            Tupla (image_paths, labels)
        """
        all_paths = []
        all_labels = []
        
        output_path = Path(output_dir)
        
        for class_id, prompt in class_prompts.items():
            print(f"\nGerando imagens para classe {class_id}: '{prompt}'")
            
            # Criar subdiretório para a classe
            class_dir = output_path / f"class_{class_id}"
            class_dir.mkdir(parents=True, exist_ok=True)
            
            # Gerar imagens
            paths = self.generate(
                n_samples=n_samples_per_class,
                prompts=[prompt],
                output_dir=str(class_dir),
                **kwargs
            )
            
            all_paths.extend(paths)
            all_labels.extend([class_id] * n_samples_per_class)
        
        return all_paths, all_labels
    
    def image_to_image(
        self,
        init_image: Union[str, Image.Image],
        prompt: str,
        strength: float = 0.75,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        output_path: Optional[str] = None
    ) -> Image.Image:
        """
        Geração image-to-image (variação de imagem existente)
        
        Args:
            init_image: Imagem inicial (path ou Image)
            prompt: Prompt de texto
            strength: Força da transformação (0.0-1.0)
            num_inference_steps: Steps de inferência
            guidance_scale: Escala de guidance
            output_path: Caminho para salvar (opcional)
            
        Returns:
            Imagem gerada
        """
        try:
            from diffusers import StableDiffusionImg2ImgPipeline
        except ImportError:
            raise ImportError("Img2Img não disponível")
        
        # Carregar pipeline img2img se necessário
        if not hasattr(self, 'img2img_pipe'):
            self.img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype
            ).to(self.device)
        
        # Carregar imagem
        if isinstance(init_image, str):
            init_image = Image.open(init_image).convert("RGB")
        
        # Gerar
        result = self.img2img_pipe(
            prompt=prompt,
            image=init_image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=self.generator
        )
        
        image = result.images[0]
        
        if output_path:
            image.save(output_path)
        
        return image
    
    def get_generator_info(self) -> Dict:
        """Retorna informações sobre o gerador"""
        info = super().get_generator_info()
        info.update({
            'model_id': self.model_id,
            'device': self.device,
            'dtype': str(self.dtype),
            'is_loaded': self.pipe is not None
        })
        return info


class DreamBoothGenerator(StableDiffusionGenerator):
    """
    Gerador usando DreamBooth para fine-tuning de Stable Diffusion
    
    Permite treinar o modelo em poucas imagens de uma classe específica
    para gerar variações.
    
    NOTA: Esta é uma implementação simplificada. Fine-tuning completo
    requer mais recursos e código especializado.
    """
    
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-2-1",
        device: str = "cuda",
        random_seed: int = 42
    ):
        super().__init__(model_id, device, random_seed=random_seed)
        self.generator_name = "DreamBooth"
    
    def fit(
        self,
        images: List[str],
        instance_prompt: str,
        class_prompt: Optional[str] = None,
        num_epochs: int = 100
    ):
        """
        Fine-tune com DreamBooth (implementação futura)
        
        Args:
            images: Lista de imagens de exemplo
            instance_prompt: Prompt da instância (ex: "a photo of sks dog")
            class_prompt: Prompt da classe (ex: "a photo of a dog")
            num_epochs: Número de épocas
        """
        print("⚠️ DreamBooth fine-tuning ainda não implementado")
        print("Use o modelo base para geração por enquanto")
        
        # Armazenar para referência
        self.training_images = images
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt
